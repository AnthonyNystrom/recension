"""Tests for the Budget dataclass."""

from __future__ import annotations

import pytest

from recension import Budget


def test_defaults() -> None:
    b = Budget()
    assert b.candidates_per_round == 4
    assert b.rounds == 3
    assert b.diagnosis_depth == 1
    assert b.max_model_calls is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"candidates_per_round": 0},
        {"rounds": 0},
        {"diagnosis_depth": 0},
        {"max_model_calls": 0},
    ],
)
def test_invalid_values_raise(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        Budget(**kwargs)


def test_dict_roundtrip() -> None:
    b = Budget(candidates_per_round=2, rounds=5, diagnosis_depth=3, max_model_calls=100)
    assert Budget.from_dict(b.to_dict()) == b
