import json
import threading
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sqlalchemy.orm import Session, sessionmaker

from evalkit.datasets import (
    MAX_LINE_BYTES,
    Case,
    canonical_json,
    content_hash,
    get_dataset,
    import_dataset,
    list_datasets,
    load_cases,
    parse_jsonl,
    to_jsonl,
)
from evalkit.errors import InvalidInputError, NotFoundError

SAMPLE = """
{"id": "q1", "input": {"question": "2+2?"}, "expected": "4", "tags": ["math"], "split": "test"}
{"id": "q2", "input": "Capital of France?", "expected": "Paris", "tags": ["geo", "easy"]}

{"id": "q3", "input": {"question": "3*3?"}, "expected": "9", "tags": ["math", "easy"], \
"split": "dev", "metadata": {"source": "hand"}}
"""


def test_parse_valid_jsonl() -> None:
    cases = parse_jsonl(SAMPLE)
    assert [c.id for c in cases] == ["q1", "q2", "q3"]
    assert cases[0].input == {"question": "2+2?"}
    assert cases[1].split is None
    assert cases[2].metadata == {"source": "hand"}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (
            '{"id": "a", "input": "x"}\n{"id": "a", "input": "y"}',
            "line 2: duplicate id 'a' (first on line 1)",
        ),
        (
            '{"id": "a", "input": "x", "colour": 1}',
            "line 1: colour: Extra inputs are not permitted",
        ),
        ('{"id": "a"}', "line 1: input: Field required"),
        ('{"id": "a", "input": ""}', "input must not be empty"),
        ('{"id": "a", "input": {}}', "input must not be empty"),
        ('{"id": "", "input": "x"}', "line 1: id"),
        ('{"id": "a\\u0000", "input": "x"}', "line 1: id"),
        ('{"id": "a", "input": "x", "tags": ["t", "t"]}', "tags must be unique"),
        ('{"id": "a", "input": "x", "tags": [""]}', "line 1: tags.0"),
        ('{"id": "a", "input": NaN}', "not valid JSON"),
        ("[1, 2]", "line 1: each line must be a JSON object"),
        ("{oops", "line 1: not valid JSON"),
        ("\n\n", "dataset has no cases"),
        ('{"id": "a", "input": "x", "split": 5, "tags": 3}', "(and 1 more)"),
    ],
)
def test_parse_errors_name_the_line(text: str, message: str) -> None:
    with pytest.raises(InvalidInputError) as info:
        parse_jsonl(text)
    assert message in info.value.message


def test_limits() -> None:
    long_line = json.dumps({"id": "a", "input": "x" * MAX_LINE_BYTES})
    with pytest.raises(InvalidInputError, match="longer than"):
        parse_jsonl(long_line)
    many = "\n".join(json.dumps({"id": str(i), "input": "x"}) for i in range(4))
    with pytest.raises(InvalidInputError, match="line 4: more than 3 cases"):
        parse_jsonl(many, max_cases=3)
    deep = '{"id": "a", "input": ' + "[" * 100_000 + "]" * 100_000 + "}"
    with pytest.raises(InvalidInputError, match="not valid JSON"):
        parse_jsonl(deep)


def test_bytes_input_and_export_round_trip() -> None:
    cases = parse_jsonl(SAMPLE.encode())
    exported = to_jsonl(cases)
    again = parse_jsonl(exported)
    assert to_jsonl(again) == exported
    assert content_hash(again) == content_hash(cases)


json_scalars = st.none() | st.booleans() | st.integers(-(10**6), 10**6) | st.text(max_size=10)
json_values = st.recursive(
    json_scalars,
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(max_size=5), children, max_size=3)
    ),
    max_leaves=8,
)
cases_strategy = st.lists(
    st.builds(
        Case,
        id=st.text(alphabet="abcdefghij0123456789", min_size=1, max_size=8),
        input=st.dictionaries(st.text(min_size=1, max_size=5), json_values, min_size=1, max_size=4),
        expected=json_values,
        tags=st.lists(st.sampled_from(["a", "b", "c"]), unique=True).map(tuple),
        split=st.none() | st.sampled_from(["dev", "test"]),
    ),
    min_size=1,
    max_size=6,
    unique_by=lambda c: c.id,
)


def _shuffle_keys(value: Any, rnd: st.DataObject) -> Any:
    if isinstance(value, dict):
        keys = rnd.draw(st.permutations(list(value)))
        return {k: _shuffle_keys(value[k], rnd) for k in keys}
    if isinstance(value, list):
        return [_shuffle_keys(v, rnd) for v in value]
    return value


@given(cases_strategy, st.data())
@settings(max_examples=150)
def test_hash_ignores_key_order_whitespace_and_line_order(
    cases: list[Case], data: st.DataObject
) -> None:
    lines = []
    for case in data.draw(st.permutations(cases)):
        raw = _shuffle_keys(case.model_dump(mode="json"), data)
        indent = data.draw(st.sampled_from([None, 0, 2]))
        lines.append(json.dumps(raw, indent=indent).replace("\n", " "))
    reparsed = parse_jsonl("\n".join(lines))
    assert content_hash(reparsed) == content_hash(cases)


@given(cases_strategy, st.data())
@settings(max_examples=150)
def test_hash_changes_when_any_case_changes(cases: list[Case], data: st.DataObject) -> None:
    index = data.draw(st.integers(0, len(cases) - 1))
    field = data.draw(st.sampled_from(["input", "expected", "tags", "split", "metadata"]))
    original = cases[index]
    replacement = {
        "input": {"changed": canonical_json(original.input)},
        "expected": ["changed", original.expected],
        "tags": (*original.tags, "zz-new"),
        "split": "other" if original.split != "other" else "dev",
        "metadata": {"changed": True, **original.metadata},
    }[field]
    changed = [*cases]
    changed[index] = original.model_copy(update={field: replacement})
    assert content_hash(changed) != content_hash(cases)


def test_import_is_versioned_and_idempotent(sessions: sessionmaker[Session]) -> None:
    cases = parse_jsonl(SAMPLE)
    with sessions() as s:
        v1, created = import_dataset(s, "qa", cases, "first")
        assert created
        assert (v1.version, v1.case_count) == (1, 3)
        same, created = import_dataset(s, "qa", list(reversed(cases)))
        assert not created
        assert same.id == v1.id
        v2, created = import_dataset(s, "qa", cases[:2])
        assert created
        assert v2.version == 2
        other, _ = import_dataset(s, "other", cases)
        assert other.version == 1
    with sessions() as s:
        assert get_dataset(s, "qa").version == 2
        assert get_dataset(s, "qa", 1).description == "first"
        assert [(d.name, d.version) for d in list_datasets(s)] == [
            ("other", 1),
            ("qa", 1),
            ("qa", 2),
        ]
        with pytest.raises(NotFoundError, match="'qa' version 9"):
            get_dataset(s, "qa", 9)
        with pytest.raises(NotFoundError, match="'nope' not found"):
            get_dataset(s, "nope")


def test_load_cases_keeps_order_and_filters(sessions: sessionmaker[Session]) -> None:
    cases = parse_jsonl(SAMPLE)
    with sessions() as s:
        ds, _ = import_dataset(s, "qa", cases)
        assert load_cases(s, ds) == cases
        assert [c.id for c in load_cases(s, ds, split="test")] == ["q1"]
        assert [c.id for c in load_cases(s, ds, tags=["easy"])] == ["q2", "q3"]
        assert [c.id for c in load_cases(s, ds, tags=["easy", "math"])] == ["q3"]
        assert load_cases(s, ds, split="dev", tags=["geo"]) == []


@pytest.mark.parametrize("name", ["", "-x", "a b", "a/b", "x" * 129])
def test_bad_names(sessions: sessionmaker[Session], name: str) -> None:
    with sessions() as s, pytest.raises(InvalidInputError, match="dataset name"):
        import_dataset(s, name, parse_jsonl(SAMPLE))


def test_long_description(sessions: sessionmaker[Session]) -> None:
    with sessions() as s, pytest.raises(InvalidInputError, match="description"):
        import_dataset(s, "qa", parse_jsonl(SAMPLE), "x" * 2001)


def test_concurrent_imports_of_same_content_make_one_version(
    sessions: sessionmaker[Session],
) -> None:
    cases = parse_jsonl(SAMPLE)
    results: list[tuple[int, bool]] = []
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with sessions() as s:
                ds, created = import_dataset(s, "race", cases)
                results.append((ds.version, created))
        except BaseException as exc:  # pragma: no cover - surfaced by the assert below
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert {v for v, _ in results} == {1}
    assert sum(created for _, created in results) == 1


def test_version_allocation_gives_up_after_repeated_conflicts(
    sessions: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    from sqlalchemy.exc import IntegrityError

    def always_conflict(self: Session) -> None:
        raise IntegrityError("insert", {}, Exception("conflict"))

    with sessions() as s:
        monkeypatch.setattr(Session, "commit", always_conflict)
        with pytest.raises(InvalidInputError, match="could not allocate"):
            import_dataset(s, "qa", parse_jsonl(SAMPLE))


def test_non_utf8_is_rejected() -> None:
    with pytest.raises(InvalidInputError, match=r"not UTF-8 \(byte 8\)"):
        parse_jsonl(b'{"id": "\xff"}')
