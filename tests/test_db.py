from datetime import UTC, datetime

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, inspect
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from evalkit.db import make_engine, upgrade
from evalkit.db.models import Base, Dataset, Result, Run, as_utc
from evalkit.db.session import alembic_config


def test_migration_matches_models(engine: Engine) -> None:
    with engine.connect() as conn:
        ctx = MigrationContext.configure(conn, opts={"compare_type": True})
        assert compare_metadata(ctx, Base.metadata) == []
    tables = set(inspect(engine).get_table_names())
    assert {"datasets", "dataset_cases", "runs", "results", "response_cache"} <= tables


def test_downgrade_removes_everything(engine: Engine) -> None:
    from alembic import command

    command.downgrade(alembic_config(engine), "base")
    assert set(inspect(engine).get_table_names()) == {"alembic_version"}
    upgrade(engine)
    assert "runs" in inspect(engine).get_table_names()


def test_env_requires_engine() -> None:
    from alembic import command
    from alembic.config import Config

    from evalkit.db.session import MIGRATIONS

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    with pytest.raises(RuntimeError, match="supplies the engine"):
        command.upgrade(cfg, "head")


def _run(session: Session) -> Run:
    ds = Dataset(name="d", version=1, content_hash="h" * 64, case_count=0)
    session.add(ds)
    session.flush()
    run = Run(
        id="r1",
        dataset_id=ds.id,
        target_spec={},
        target_hash="t" * 64,
        scorers=[],
        options={},
        seed=1,
    )
    session.add(run)
    session.flush()
    return run


def test_result_per_run_and_case_is_unique(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        run = _run(s)
        s.add(Result(run_id=run.id, case_id="c1", status="ok"))
        s.commit()
        s.add(Result(run_id=run.id, case_id="c1", status="ok"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_foreign_keys_are_enforced_on_sqlite(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        s.add(Result(run_id="missing", case_id="c1", status="ok"))
        with pytest.raises(IntegrityError):
            s.commit()


def test_defaults_and_utc(sessions: sessionmaker[Session]) -> None:
    with sessions() as s:
        run = _run(s)
        s.commit()
        loaded = s.get(Run, run.id)
        assert loaded is not None
        assert loaded.status == "pending"
        assert as_utc(loaded.created_at).tzinfo is UTC
    aware = datetime(2020, 1, 1, tzinfo=UTC)
    assert as_utc(aware) is aware
    assert as_utc(datetime(2020, 1, 1)).tzinfo is UTC


def test_in_memory_sqlite_and_postgres_engines() -> None:
    mem = make_engine("sqlite://")
    upgrade(mem)
    assert "runs" in inspect(mem).get_table_names()
    mem.dispose()
    pg = make_engine("postgresql+psycopg://user@localhost:1/none")
    assert pg.pool.size() == 5  # type: ignore[attr-defined]
    pg.dispose()
