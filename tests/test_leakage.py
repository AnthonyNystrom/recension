"""Tests for the leakage heuristics."""

from __future__ import annotations

from recension import EvalSet, Example
from recension.leakage import check_candidate

VALIDATION_INPUT = "The quarterly revenue figures for the northeast region were strong."


def make_evalset() -> EvalSet:
    return EvalSet(
        train=[Example(id="t1", input="short train input", expected="yes")],
        validation=[Example(id="v1", input=VALIDATION_INPUT, expected="positive")],
    )


class TestVerbatimSpan:
    def test_flags_long_copied_validation_span(self) -> None:
        candidate = f"Classify carefully. Example: {VALIDATION_INPUT}"
        flags = check_candidate(candidate, "Classify carefully.", make_evalset())
        assert len(flags) == 1
        assert flags[0].kind == "verbatim_validation_span"
        assert "v1" in flags[0].detail

    def test_span_already_in_incumbent_is_not_reflagged(self) -> None:
        text = f"Classify carefully. Example: {VALIDATION_INPUT}"
        assert check_candidate(text, text, make_evalset()) == []

    def test_short_overlap_is_not_flagged(self) -> None:
        candidate = "Classify the quarterly revenue text."  # < min span, common words
        assert check_candidate(candidate, "Classify.", make_evalset()) == []

    def test_clean_candidate_has_no_flags(self) -> None:
        assert check_candidate("Label the sentiment.", "Label it.", make_evalset()) == []

    def test_flag_renders_with_kind_prefix(self) -> None:
        candidate = f"x {VALIDATION_INPUT}"
        (flag,) = check_candidate(candidate, "x", make_evalset())
        assert str(flag).startswith("verbatim_validation_span: ")


class TestImplausibleGain:
    def test_validation_jump_with_flat_train_is_flagged(self) -> None:
        flags = check_candidate(
            "a", "b", make_evalset(), train_gain=0.0, validation_gain=0.5
        )
        assert [f.kind for f in flags] == ["implausible_gain"]

    def test_negative_train_gain_is_flagged(self) -> None:
        flags = check_candidate(
            "a", "b", make_evalset(), train_gain=-0.1, validation_gain=0.3
        )
        assert [f.kind for f in flags] == ["implausible_gain"]

    def test_disproportionate_ratio_is_flagged(self) -> None:
        flags = check_candidate(
            "a", "b", make_evalset(), train_gain=0.05, validation_gain=0.4
        )
        assert [f.kind for f in flags] == ["implausible_gain"]

    def test_proportional_gain_is_clean(self) -> None:
        assert (
            check_candidate("a", "b", make_evalset(), train_gain=0.3, validation_gain=0.35)
            == []
        )

    def test_small_validation_gain_is_noise_not_leakage(self) -> None:
        assert (
            check_candidate("a", "b", make_evalset(), train_gain=0.0, validation_gain=0.05)
            == []
        )

    def test_no_gains_supplied_skips_check(self) -> None:
        assert check_candidate("a", "b", make_evalset()) == []
