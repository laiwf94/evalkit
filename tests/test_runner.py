import asyncio
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, sessionmaker

from evalkit import leases
from evalkit.config import Settings
from evalkit.datasets import Case, import_dataset
from evalkit.db.models import CachedResponse, CaseLease, Result, Run, utcnow
from evalkit.errors import ConflictError, InvalidInputError, NotFoundError, PreconditionError
from evalkit.redaction import PatternSpec, RedactionConfig
from evalkit.report import compare_runs, summarize
from evalkit.runner import CaseResult, ExecutionReport, Runner
from evalkit.runspec import RunConfig
from evalkit.targets import Target
from evalkit.targets.fake import FakeTarget


def make_cases(n: int = 40) -> list[Case]:
    return [
        Case(
            id=f"c{i:03d}",
            input={"q": f"question {i}"},
            expected=f"answer {i}",
            split="test" if i % 2 else "dev",
        )
        for i in range(n)
    ]


@pytest.fixture
def dataset(sessions: sessionmaker[Session]) -> str:
    with sessions() as s:
        import_dataset(s, "qa", make_cases())
    return "qa"


def config(dataset: str = "qa", **target: Any) -> RunConfig:
    options = target.pop("options", {})
    scorers = target.pop("scorers", [{"type": "exact"}, {"type": "token_f1", "gate": False}])
    run_filter = target.pop("filter", {})
    return RunConfig.model_validate(
        {
            "label": "test",
            "dataset": {"name": dataset},
            "target": {"type": "fake", **target},
            "scorers": scorers,
            "options": {"concurrency": 4, **options},
            "filter": run_filter,
        }
    )


class Tracking(FakeTarget):
    finished = 0

    async def generate(self, case: Case) -> Any:
        gen = await super().generate(case)
        self.finished += 1
        return gen


class CountingFactory:
    def __init__(self) -> None:
        self.targets: list[Tracking] = []

    def __call__(self, spec: Any) -> Target:
        t = Tracking(spec)
        self.targets.append(t)
        return t

    @property
    def calls(self) -> int:
        return sum(t.calls for t in self.targets)

    @property
    def finished(self) -> int:
        return sum(t.finished for t in self.targets)


def runner(sessions: sessionmaker[Session], settings: Settings | None = None, **kw: Any) -> Runner:
    return Runner(sessions, settings or Settings.from_env({}), allow_import_paths=True, **kw)


def result_count(sessions: sessionmaker[Session], run_id: str) -> int:
    with sessions() as s:
        return (
            s.scalar(select(func.count()).select_from(Result).where(Result.run_id == run_id)) or 0
        )


def test_run_records_every_case_and_summarizes(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    r = runner(sessions)
    run = r.create_run(config(accuracy=0.75, seed=1))
    report = asyncio.run(r.execute(run.id))
    assert (report.executed, report.cached, report.errors, report.skipped) == (40, 0, 0, 0)
    with sessions() as s:
        summary = summarize(s, run.id, resamples=500)
        stored = s.get_one(Run, run.id)
        assert stored.lease_owner is None
        assert stored.finished_at is not None
    assert summary.status == "completed"
    assert (summary.cases_total, summary.completed, summary.ok, summary.errors) == (40, 40, 40, 0)
    exact = summary.metrics[0]
    assert exact.name == "exact"
    assert exact.gate
    assert not summary.metrics[1].gate
    assert 0.5 < (exact.mean or 0) < 0.95
    assert exact.ci_low is not None
    assert exact.ci_high is not None
    assert exact.ci_low <= exact.mean <= exact.ci_high  # type: ignore[operator]
    assert summary.input_tokens > 0
    assert summary.latency_p50_ms is not None
    assert summary.target_type == "fake"


def test_rerun_is_served_from_cache_for_free(sessions: sessionmaker[Session], dataset: str) -> None:
    factory = CountingFactory()
    r = runner(sessions, target_factory=factory)
    first = r.create_run(config(accuracy=0.5))
    asyncio.run(r.execute(first.id))
    calls_after_first = factory.calls
    second = r.create_run(config(accuracy=0.5))
    report = asyncio.run(r.execute(second.id))
    assert calls_after_first == 40
    assert factory.calls == 40
    assert (report.executed, report.cached) == (0, 40)
    with sessions() as s:
        a, b = summarize(s, first.id, resamples=200), summarize(s, second.id, resamples=200)
    assert b.cost_usd == 0.0
    assert b.input_tokens == 0
    assert b.cached == 40
    assert a.metrics[0].mean == b.metrics[0].mean
    assert b.latency_p50_ms is None


def test_cache_can_be_disabled_and_is_keyed_by_target(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = CountingFactory()
    r = runner(sessions, target_factory=factory)
    asyncio.run(r.execute(r.create_run(config(accuracy=0.5)).id))
    asyncio.run(r.execute(r.create_run(config(accuracy=0.5, options={"use_cache": False})).id))
    asyncio.run(r.execute(r.create_run(config(accuracy=0.6)).id))
    assert factory.calls == 120


def test_filter_limits_cases(sessions: sessionmaker[Session], dataset: str) -> None:
    r = runner(sessions)
    run = r.create_run(config(filter={"split": "test"}))
    assert asyncio.run(r.execute(run.id)).executed == 20
    with sessions() as s:
        assert summarize(s, run.id, resamples=200).cases_total == 20


def test_interrupted_run_resumes_without_repeating_cases(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = CountingFactory()
    r = runner(sessions, target_factory=factory)
    run = r.create_run(config(latency_ms=30, options={"concurrency": 2}))

    async def interrupt() -> None:
        task = asyncio.create_task(r.execute(run.id))
        while result_count(sessions, run.id) < 6:
            await asyncio.sleep(0.02)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(interrupt())
    # Every answer the target returned before the interrupt was saved.
    assert result_count(sessions, run.id) == factory.finished
    with sessions() as s:
        assert s.get_one(Run, run.id).status == "interrupted"
        assert s.get_one(Run, run.id).lease_owner is None
    done_before = result_count(sessions, run.id)
    assert 6 <= done_before < 40

    report = asyncio.run(runner(sessions, target_factory=factory).execute(run.id))
    assert report.skipped == done_before
    assert report.executed == 40 - done_before
    with sessions() as s:
        ids = list(s.scalars(select(Result.case_id).where(Result.run_id == run.id)))
    assert sorted(ids) == sorted({c.id for c in make_cases()})
    assert len(ids) == len(set(ids))


def test_completed_run_is_not_executed_again(sessions: sessionmaker[Session], dataset: str) -> None:
    r = runner(sessions)
    run = r.create_run(config())
    asyncio.run(r.execute(run.id))
    assert asyncio.run(runner(sessions).execute(run.id)).executed == 0


def test_two_processes_share_a_run(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        import_dataset(s, "big", make_cases(400))
    factory = CountingFactory()
    a = runner(sessions, owner="a", target_factory=factory)
    b = runner(sessions, owner="b", target_factory=factory)
    run = a.create_run(config("big", latency_ms=2, options={"concurrency": 4}))

    async def both() -> tuple[ExecutionReport, ExecutionReport]:
        return await asyncio.gather(a.execute(run.id), b.execute(run.id))

    first, second = asyncio.run(both())
    assert first.executed > 0
    assert second.executed > 0
    assert first.executed + second.executed == factory.calls == 400
    assert first.refused == second.refused == 0
    with sessions() as s:
        ids = list(s.scalars(select(Result.case_id).where(Result.run_id == run.id)))
        stored = s.get_one(Run, run.id)
        assert stored.status == "completed"
        assert stored.finished_at is not None
        assert leases.remaining(s, run.id) == 0
    assert sorted(ids) == sorted(c.id for c in make_cases(400))


def test_cases_held_by_a_dead_process_are_taken_over(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    ttl = timedelta(seconds=0.5)
    dead = runner(sessions, owner="dead", lease_ttl=ttl)
    run = dead.create_run(config())
    with sessions() as s:
        leases.seed(s, run.id, [c.id for c in make_cases()])
        s.commit()
    held = dead._claim_cases(run.id, 10)
    assert len(held) == 10
    # The survivor runs the other 30, waits for the dead process's leases to
    # expire, and then runs those too.
    survivor = runner(sessions, owner="survivor", lease_ttl=ttl, heartbeat_seconds=0.1)
    report = asyncio.run(survivor.execute(run.id))
    assert report.executed == 40
    assert result_count(sessions, run.id) == 40
    with sessions() as s:
        assert s.get_one(Run, run.id).status == "completed"
        fences = {row.case_id: row.fence for row in s.scalars(select(CaseLease))}
    assert {fences[c.case_id] for c in held} == {2}


def test_waiting_for_another_process(sessions: sessionmaker[Session], dataset: str) -> None:
    r = runner(sessions, owner="a", heartbeat_seconds=2)
    run = r.create_run(config())
    with sessions() as s:
        leases.seed(s, run.id, ["c000", "c001"])
        s.commit()

    def wait() -> float:
        return r._recheck_after(r._outstanding(run.id))

    assert wait() == 0.05  # unowned, but not claimed: look again soon
    runner(sessions, owner="b")._claim_cases(run.id, 2)
    assert wait() == 2  # live leases: one heartbeat
    with sessions() as s:
        s.execute(update(CaseLease).values(expires_at=utcnow() + timedelta(seconds=0.5)))
        s.commit()
    assert 0.05 <= wait() <= 0.5  # until the first lease could expire
    with sessions() as s:
        s.execute(update(CaseLease).values(expires_at=utcnow() - timedelta(seconds=5)))
        s.commit()
    assert wait() == 0.05  # already expired: claim it now


def test_unknown_run(sessions: sessionmaker[Session], dataset: str) -> None:
    with pytest.raises(NotFoundError):
        asyncio.run(runner(sessions).execute("nope"))


def test_results_of_cases_taken_over_are_refused(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = CountingFactory()
    r = runner(sessions, owner="a", target_factory=factory)
    run = r.create_run(config(latency_ms=20, options={"concurrency": 2}))

    async def steal() -> Any:
        task = asyncio.create_task(r.execute(run.id))
        while result_count(sessions, run.id) < 1:
            await asyncio.sleep(0.01)
        # Another process takes over every case "a" holds (as if a's leases had
        # expired), and itself dies straight away, so "a" ends up re-claiming them.
        with sessions() as s:
            s.execute(
                update(CaseLease)
                .where(CaseLease.owner == "a", CaseLease.done.is_(False))
                .values(
                    owner="thief",
                    fence=CaseLease.fence + 1,
                    expires_at=utcnow() - timedelta(seconds=1),
                )
            )
            s.commit()
        return await task

    report = asyncio.run(steal())
    # Results computed under the old claims were refused, and those cases ran again.
    assert report.refused > 0
    assert report.executed == 40 + report.refused == factory.calls
    with sessions() as s:
        ids = list(s.scalars(select(Result.case_id).where(Result.run_id == run.id)))
        assert s.get_one(Run, run.id).status == "completed"
    assert sorted(ids) == sorted(c.id for c in make_cases())


def test_target_failures_become_error_results(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    r = runner(sessions)
    run = r.create_run(config(failure_rate=0.95, options={"retry": {"max_attempts": 1}}))
    report = asyncio.run(r.execute(run.id))
    assert report.errors > 20
    with sessions() as s:
        failed = s.scalars(
            select(Result).where(Result.run_id == run.id, Result.status == "error")
        ).first()
        assert failed is not None
        assert failed.error == "fake transient failure"
        assert failed.scores == {}
        assert failed.attempts == 1
        summary = summarize(s, run.id, resamples=200)
    # Errors score 0, so the mean is at most the success rate.
    assert (summary.metrics[0].mean or 0) <= summary.ok / 40


def test_retries_recover_transient_failures(sessions: sessionmaker[Session], dataset: str) -> None:
    r = runner(sessions)
    run = r.create_run(
        config(failure_rate=0.5, options={"retry": {"max_attempts": 10, "base_delay": 0}})
    )
    assert asyncio.run(r.execute(run.id)).errors == 0
    with sessions() as s:
        attempts = list(s.scalars(select(Result.attempts).where(Result.run_id == run.id)))
    assert max(attempts) > 1


def test_template_errors_and_timeouts_are_per_case(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    r = runner(sessions)
    run = r.create_run(config(latency_ms=300, options={"case_timeout_seconds": 0.05}))
    asyncio.run(r.execute(run.id))
    with sessions() as s:
        errors = set(s.scalars(select(Result.error).where(Result.run_id == run.id)))
    assert errors == {"case took longer than 0.05 s"}

    from evalkit.errors import InvalidInputError as Bad

    class BadTemplate(FakeTarget):
        async def generate(self, case: Case) -> Any:
            raise Bad("case input has no field 'x' used by the template")

    r2 = runner(sessions, target_factory=BadTemplate)
    run2 = r2.create_run(config(seed=9))
    assert asyncio.run(r2.execute(run2.id)).errors == 40


def test_scorer_bugs_and_out_of_range_values_are_unscorable(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    r = runner(sessions)
    run = r.create_run(
        config(
            scorers=[
                {"type": "plugin", "name": "boom", "entry": "tests.plugin_scorer:Exploding"},
                {"type": "plugin", "name": "big", "entry": "tests.plugin_scorer:OutOfRange"},
                {
                    "type": "plugin",
                    "name": "nan",
                    "entry": "tests.plugin_scorer:OutOfRange",
                    "options": {"value": "nan"},
                },
            ]
        )
    )
    asyncio.run(r.execute(run.id))
    with sessions() as s:
        row = s.scalars(select(Result).where(Result.run_id == run.id)).first()
        assert row is not None
        assert row.scores["boom"] == {"value": None, "detail": "scorer failed: plugin bug"}
        assert row.scores["big"] == {"value": None, "detail": "scorer returned 2.0, outside [0, 1]"}
        assert row.scores["nan"]["value"] is None
        summary = summarize(s, run.id, resamples=200)
    assert summary.metrics[0].mean is None
    assert summary.metrics[0].unscorable == 40


def test_outputs_are_redacted_before_storage(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        import_dataset(s, "pii", [Case(id="p1", input="who?", expected="mail alice@example.com")])
    r = runner(sessions)
    run = r.create_run(config("pii", scorers=[{"type": "exact"}]))
    asyncio.run(r.execute(run.id))
    with sessions() as s:
        row = s.scalars(select(Result).where(Result.run_id == run.id)).one()
        cached = s.scalars(select(CachedResponse)).one()
    assert row.output == "mail [EMAIL]"
    assert cached.output == "mail [EMAIL]"
    # The scorer saw the redacted text, so it does not match the raw expected value.
    assert row.scores["exact"]["value"] == 0.0


def test_failed_redaction_is_not_cached(sessions: sessionmaker[Session], dataset: str) -> None:
    settings = Settings(
        redaction=RedactionConfig(
            builtins=(),
            patterns=(PatternSpec(name="BAD", pattern=r"(a|aa)+$"),),
            timeout_seconds=0.01,
        )
    )

    class Slow(FakeTarget):
        async def generate(self, case: Case) -> Any:
            gen = await super().generate(case)
            return type(gen)("a" * 40 + "!", gen.stop_reason, gen.usage)

    r = runner(sessions, settings, target_factory=Slow)
    run = r.create_run(config(filter={"split": "test"}))
    asyncio.run(r.execute(run.id))
    with sessions() as s:
        outputs = set(s.scalars(select(Result.output).where(Result.run_id == run.id)))
        assert s.scalar(select(func.count()).select_from(CachedResponse)) == 0
    assert outputs == {"[REDACTION_FAILED]"}


def test_unexpected_errors_fail_the_run(sessions: sessionmaker[Session], dataset: str) -> None:
    class Broken(FakeTarget):
        async def generate(self, case: Case) -> Any:
            raise ZeroDivisionError("bug in target")

    r = runner(sessions, target_factory=Broken)
    run = r.create_run(config())
    with pytest.raises(ZeroDivisionError):
        asyncio.run(r.execute(run.id))
    with sessions() as s:
        stored = s.get_one(Run, run.id)
    assert stored.status == "failed"
    assert stored.error == "ZeroDivisionError: bug in target"


def test_create_run_validation(sessions: sessionmaker[Session], dataset: str) -> None:
    r = runner(sessions)
    with pytest.raises(NotFoundError):
        r.create_run(
            config().model_copy(
                update={"dataset": config().dataset.model_copy(update={"name": "missing"})}
            )
        )
    with pytest.raises(InvalidInputError, match="selects no cases"):
        r.create_run(config(filter={"split": "nope"}))
    with pytest.raises(InvalidInputError, match="batch mode"):
        r.create_run(config(options={"mode": "batch"}))
    http = RunConfig.model_validate(
        {
            "dataset": {"name": "qa"},
            "target": {"type": "http", "endpoint": "nope"},
            "scorers": [{"type": "exact"}],
        }
    )
    with pytest.raises(InvalidInputError, match="unknown HTTP endpoint 'nope'"):
        r.create_run(http)
    with pytest.raises(InvalidInputError, match="repeated"):
        r.create_run(config(scorers=[{"type": "exact"}, {"type": "exact"}]))


def test_duplicate_results_are_skipped_not_fatal(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    r = runner(sessions)
    run = r.create_run(config())
    with sessions() as s:
        leases.seed(s, run.id, ["c000", "c001", "c002", "c003"])
        s.commit()
    fence = {c.case_id: c.fence for c in r._claim_cases(run.id, 4)}

    def result(case_id: str, output: str) -> CaseResult:
        return CaseResult(case_id=case_id, status="ok", output=output, fence=fence[case_id])

    assert r._write(run.id, [result("c000", "x")]) == 0
    # A second result for a finished case is refused by its lease...
    assert r._write(run.id, [result("c000", "dup"), result("c001", "y")]) == 1
    # ...and one written outside the lease protocol is kept, not overwritten.
    with sessions() as s:
        s.add(Result(run_id=run.id, case_id="c002", status="ok", output="old"))
        s.commit()
    batch = [result("c000", "dup"), result("c002", "new"), result("c003", "z")]
    assert r._write(run.id, batch) == 2
    with sessions() as s:
        query = select(Result.case_id, Result.output).where(Result.run_id == run.id)
        rows = {row.case_id: row.output for row in s.execute(query)}
        assert leases.remaining(s, run.id) == 0
    assert rows == {"c000": "x", "c001": "y", "c002": "old", "c003": "z"}


def test_compare_detects_regression(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        import_dataset(s, "qa", make_cases(300))
    r = runner(sessions)
    base = r.create_run(config(accuracy=0.9, seed=1))
    worse = r.create_run(config(accuracy=0.6, seed=1))
    same = r.create_run(config(accuracy=0.9, seed=1, wrong_answer="no idea"))
    for run in (base, worse, same):
        asyncio.run(r.execute(run.id))
    with sessions() as s:
        bad = compare_runs(s, base.id, worse.id, resamples=1000)
        ok = compare_runs(s, base.id, same.id, resamples=1000)
        better = compare_runs(s, worse.id, base.id, resamples=1000)
        both = compare_runs(s, base.id, worse.id, metrics=["exact", "token_f1"], resamples=1000)
    assert not bad.passed
    assert bad.regressions == ["exact"]
    assert [c.metric for c in bad.comparisons] == ["exact"]  # token_f1 is not gated
    assert bad.comparisons[0].verdict == "worse"
    assert ok.passed
    assert ok.comparisons[0].verdict == "no detectable change"
    assert better.comparisons[0].verdict == "better"
    assert [c.metric for c in both.comparisons] == ["exact", "token_f1"]


def test_compare_preconditions(sessions: sessionmaker[Session], dataset: str) -> None:
    r = runner(sessions)
    done = r.create_run(config())
    asyncio.run(r.execute(done.id))
    pending = r.create_run(config())
    with sessions() as s:
        with pytest.raises(PreconditionError, match="not completed"):
            compare_runs(s, done.id, pending.id)
        with pytest.raises(InvalidInputError, match="not in both runs"):
            compare_runs(s, done.id, done.id, metrics=["nope"])
        import_dataset(s, "qa", make_cases(10))
    other = r.create_run(config())
    asyncio.run(r.execute(other.id))
    ungated = r.create_run(config(scorers=[{"type": "exact", "gate": False}]))
    asyncio.run(r.execute(ungated.id))
    with sessions() as s:
        with pytest.raises(PreconditionError, match="different dataset contents"):
            compare_runs(s, done.id, other.id)
        with pytest.raises(PreconditionError, match="no gated metric"):
            compare_runs(s, other.id, ungated.id)
        with pytest.raises(NotFoundError):
            compare_runs(s, "x", done.id)


def test_compare_needs_cases_scored_in_both_runs(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    r = runner(sessions)
    runs = [r.create_run(config(scorers=[{"type": "numeric"}])) for _ in range(2)]
    for run in runs:
        asyncio.run(r.execute(run.id))
    with sessions() as s, pytest.raises(PreconditionError, match="no case was scored in both"):
        compare_runs(s, runs[0].id, runs[1].id)


def test_interrupt_saves_queued_results(sessions: sessionmaker[Session], dataset: str) -> None:
    r = runner(sessions)
    run = r.create_run(config())
    with sessions() as s:
        leases.seed(s, run.id, ["c000", "c001"])
        s.commit()
    fence = {c.case_id: c.fence for c in r._claim_cases(run.id, 2)}

    async def main() -> None:
        queue: asyncio.Queue[CaseResult | None] = asyncio.Queue()
        queue.put_nowait(CaseResult(case_id="c001", status="ok", output="b", fence=fence["c001"]))
        queue.put_nowait(None)
        await r._save_unwritten(
            run.id,
            queue,
            [CaseResult(case_id="c000", status="ok", output="a", fence=fence["c000"])],
        )
        await r._save_unwritten(run.id, asyncio.Queue(), [])

    asyncio.run(main())
    assert result_count(sessions, run.id) == 2


def test_an_interrupted_process_leaves_the_status_to_the_others(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    a = runner(sessions, owner="a")
    run = a.create_run(config())
    with sessions() as s:
        leases.seed(s, run.id, [c.id for c in make_cases()])
        s.commit()
    a._join(run.id)
    a._claim_cases(run.id, 5)
    runner(sessions, owner="b")._claim_cases(run.id, 5)
    a._finish(run.id, "interrupted", "")
    with sessions() as s:
        # b still works on the run, so it stays running; a's cases are free again.
        assert s.get_one(Run, run.id).status == "running"
        assert leases.outstanding(s, run.id, "b").unowned == 35
    runner(sessions, owner="b")._finish(run.id, "interrupted", "")
    with sessions() as s:
        assert s.get_one(Run, run.id).status == "interrupted"


def test_completion_needs_every_case_finished(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    r = runner(sessions)
    run = r.create_run(config())
    with sessions() as s:
        leases.seed(s, run.id, ["c000"])
        s.commit()
    r._join(run.id)
    assert r._complete(run.id) == "running"
    (claim,) = r._claim_cases(run.id, 1)
    r._write(run.id, [CaseResult(case_id="c000", status="ok", fence=claim.fence)])
    assert r._complete(run.id) == "completed"
    # A second completer changes nothing.
    with sessions() as s:
        finished = s.get_one(Run, run.id).finished_at
    assert runner(sessions)._complete(run.id) == "completed"
    with sessions() as s:
        assert s.get_one(Run, run.id).finished_at == finished


def test_a_run_from_before_case_leases_resumes(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    r = runner(sessions)
    run = r.create_run(config())
    # Written by an older version that died: results but no case leases, and a
    # run-level lease that is no longer used for online runs.
    with sessions() as s:
        s.add_all(Result(run_id=run.id, case_id=f"c{i:03d}", status="ok") for i in range(15))
        s.execute(
            update(Run)
            .where(Run.id == run.id)
            .values(
                status="running", lease_owner="old", lease_expires_at=utcnow() + timedelta(hours=1)
            )
        )
        s.commit()
    report = asyncio.run(r.execute(run.id))
    assert (report.skipped, report.executed) == (15, 25)
    with sessions() as s:
        assert s.scalar(select(func.count()).select_from(CaseLease)) == 25
        stored = s.get_one(Run, run.id)
        assert (stored.status, stored.lease_owner) == ("completed", None)
    assert result_count(sessions, run.id) == 40


def test_judge_verdicts_are_cached_across_runs(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    import json as _json

    from tests.fake_anthropic import FakeAnthropic, json_reply, message

    fake = FakeAnthropic()
    fake.default = lambda body: json_reply(message(_json.dumps({"reasoning": "r", "score": 5})))
    r = Runner(sessions, Settings(), claude=fake.client())
    scorers = [{"type": "judge", "rubric": "be right"}]
    first = r.create_run(config(scorers=scorers, filter={"split": "test"}))
    asyncio.run(r.execute(first.id))
    calls = len(fake.requests)
    second = r.create_run(config(scorers=scorers, filter={"split": "test"}))
    asyncio.run(r.execute(second.id))
    assert calls == 20
    assert len(fake.requests) == 20
    with sessions() as s:
        a, b = summarize(s, first.id, resamples=200), summarize(s, second.id, resamples=200)
    assert a.judge_cost_usd > 0
    assert b.judge_cost_usd == 0.0
    assert b.metrics[0].mean == a.metrics[0].mean == 1.0


def test_one_process_never_executes_a_run_twice(
    sessions: sessionmaker[Session], dataset: str
) -> None:
    factory = CountingFactory()
    r = runner(sessions, target_factory=factory)
    run = r.create_run(config(latency_ms=20, options={"concurrency": 2}))

    async def both() -> Any:
        return await asyncio.gather(r.execute(run.id), r.execute(run.id), return_exceptions=True)

    first, second = asyncio.run(both())
    assert first.executed == 40
    assert isinstance(second, ConflictError)
    assert factory.calls == 40
