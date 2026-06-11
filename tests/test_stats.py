"""Tests for the seeded paired-difference bootstrap."""

from __future__ import annotations

import pytest

from recension.stats import paired_bootstrap


def test_clear_gain_is_significant() -> None:
    incumbent = [0.0] * 20
    candidate = [1.0] * 20
    result = paired_bootstrap(incumbent, candidate, seed=1)
    assert result.mean_difference == 1.0
    assert result.significant
    assert result.ci_low > 0.0


def test_no_gain_is_not_significant() -> None:
    scores = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    result = paired_bootstrap(scores, scores, seed=1)
    assert result.mean_difference == 0.0
    assert not result.significant
    assert result.ci_low <= 0.0 <= result.ci_high


def test_tiny_noisy_gain_is_not_significant() -> None:
    # One example flips; on a small noisy sample the CI includes 0.
    incumbent = [1.0, 0.0, 1.0, 0.0, 0.0]
    candidate = [1.0, 1.0, 1.0, 0.0, 0.0]
    result = paired_bootstrap(incumbent, candidate, seed=7)
    assert result.mean_difference == pytest.approx(0.2)
    assert not result.significant  # one-example edge is within noise


def test_deterministic_under_seed() -> None:
    inc = [0.0, 1.0, 0.0, 1.0, 0.0]
    cand = [1.0, 1.0, 1.0, 1.0, 0.0]
    a = paired_bootstrap(inc, cand, seed=42)
    b = paired_bootstrap(inc, cand, seed=42)
    assert a == b


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_bootstrap([0.0, 1.0], [1.0], seed=1)


def test_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        paired_bootstrap([], [], seed=1)
