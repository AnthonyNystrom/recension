"""Tests for EvalSet construction, loaders, and split integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from recension import DegenerateEvalError, EvalSet, Example


def ex(i: str, split_input: str = "hello") -> Example:
    return Example(id=i, input=split_input, expected="world")


def test_valid_construction() -> None:
    es = EvalSet(train=[ex("t1"), ex("t2")], validation=[ex("v1")])
    assert [e.id for e in es.train] == ["t1", "t2"]
    assert [e.id for e in es.validation] == ["v1"]
    assert es.test == ()  # optional, defaults to empty
    assert "train=2" in repr(es)
    assert "test=0" in repr(es)


def test_optional_test_split() -> None:
    es = EvalSet(train=[ex("t1")], validation=[ex("v1")], test=[ex("x1"), ex("x2")])
    assert [e.id for e in es.test] == ["x1", "x2"]
    assert "test=2" in repr(es)


def test_test_split_must_be_disjoint_from_others() -> None:
    with pytest.raises(DegenerateEvalError, match="train and test"):
        EvalSet(train=[ex("a")], validation=[ex("v1")], test=[ex("a")])
    with pytest.raises(DegenerateEvalError, match="validation and test"):
        EvalSet(train=[ex("t1")], validation=[ex("a")], test=[ex("a")])


def test_duplicate_ids_within_test_split_raise() -> None:
    with pytest.raises(DegenerateEvalError, match="duplicate.*test"):
        EvalSet(train=[ex("t1")], validation=[ex("v1")], test=[ex("x"), ex("x")])


def test_empty_train_raises() -> None:
    with pytest.raises(DegenerateEvalError, match="train split is empty"):
        EvalSet(train=[], validation=[ex("v1")])


def test_empty_validation_raises() -> None:
    with pytest.raises(DegenerateEvalError, match="validation split is empty"):
        EvalSet(train=[ex("t1")], validation=[])


def test_duplicate_ids_within_split_raise() -> None:
    with pytest.raises(DegenerateEvalError, match="duplicate"):
        EvalSet(train=[ex("a"), ex("a")], validation=[ex("v1")])


def test_overlapping_ids_across_splits_raise() -> None:
    with pytest.raises(DegenerateEvalError, match="both train and validation"):
        EvalSet(train=[ex("a")], validation=[ex("a")])


def test_from_records_routes_splits_and_metadata() -> None:
    es = EvalSet.from_records(
        [
            {"id": "t1", "input": "x", "expected": "y", "split": "train", "domain": "news"},
            {"id": "v1", "input": "x", "rubric": "be terse", "split": "validation"},
        ]
    )
    assert es.train[0].metadata == {"domain": "news"}
    assert es.train[0].expected == "y"
    assert es.validation[0].rubric == "be terse"
    assert es.validation[0].expected is None


def test_from_records_missing_key_raises() -> None:
    with pytest.raises(DegenerateEvalError, match="missing required key"):
        EvalSet.from_records([{"id": "a", "input": "x"}])


def test_from_records_routes_test_split() -> None:
    es = EvalSet.from_records(
        [
            {"id": "t1", "input": "x", "expected": "y", "split": "train"},
            {"id": "v1", "input": "x", "expected": "y", "split": "validation"},
            {"id": "x1", "input": "x", "expected": "y", "split": "test"},
        ]
    )
    assert [e.id for e in es.test] == ["x1"]


def test_from_records_unknown_split_raises() -> None:
    with pytest.raises(DegenerateEvalError, match="unknown split"):
        EvalSet.from_records([{"id": "a", "input": "x", "split": "holdout"}])


def test_from_jsonl_malformed_line_names_the_location(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    path.write_text(
        '{"id": "t1", "input": "a", "expected": "b", "split": "train"}\n{ bad json\n',
        encoding="utf-8",
    )
    with pytest.raises(DegenerateEvalError, match="line 2: invalid JSON"):
        EvalSet.from_jsonl(path)


def test_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "data.jsonl"
    rows = [
        {"id": "t1", "input": "a", "expected": "b", "split": "train"},
        {"id": "v1", "input": "c", "expected": "d", "split": "validation"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    es = EvalSet.from_jsonl(path)
    assert len(es.train) == 1
    assert len(es.validation) == 1
    assert es.validation[0].expected == "d"
