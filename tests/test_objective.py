"""Tests for the shipped objectives: ExactMatch, F1, LLMJudge."""

from __future__ import annotations

import pytest

from recension import F1, DegenerateEvalError, ExactMatch, Example, LLMJudge
from recension.models.base import Message


def ex(expected: str | None = "positive", rubric: str | None = None) -> Example:
    return Example(id="e1", input="great product", expected=expected, rubric=rubric)


class ScriptedModel:
    """Minimal in-test Model: returns a fixed reply, counts calls."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self._calls = 0

    @property
    def call_count(self) -> int:
        return self._calls

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        self._calls += 1
        return self.reply


class TestExactMatch:
    def test_match_is_case_and_whitespace_insensitive(self) -> None:
        assert ExactMatch().score("  Positive \n", ex()) == 1.0

    def test_mismatch(self) -> None:
        assert ExactMatch().score("negative", ex()) == 0.0

    def test_case_sensitive_mode(self) -> None:
        assert ExactMatch(case_sensitive=True).score("Positive", ex()) == 0.0
        assert ExactMatch(case_sensitive=True).score("positive", ex()) == 1.0

    def test_missing_expected_raises(self) -> None:
        with pytest.raises(DegenerateEvalError, match="no 'expected'"):
            ExactMatch().score("x", ex(expected=None))

    def test_aggregate_mean_and_empty(self) -> None:
        assert ExactMatch().aggregate([1.0, 0.0]) == 0.5
        with pytest.raises(DegenerateEvalError, match="empty"):
            ExactMatch().aggregate([])

    def test_not_model_graded(self) -> None:
        assert ExactMatch().model_graded is False


class TestF1:
    def test_perfect_overlap(self) -> None:
        assert F1().score("positive", ex()) == 1.0

    def test_partial_overlap(self) -> None:
        example = Example(id="e", input="q", expected="the cat sat")
        score = F1().score("the cat ran", example)
        assert score == pytest.approx(2 / 3)

    def test_no_overlap(self) -> None:
        assert F1().score("zebra", ex()) == 0.0

    def test_both_empty(self) -> None:
        assert F1().score("", ex(expected="")) == 1.0

    def test_one_empty(self) -> None:
        assert F1().score("", ex()) == 0.0


class TestLLMJudge:
    def test_parses_grade_and_normalizes(self) -> None:
        judge = LLMJudge(ScriptedModel("8"), rubric="clarity")
        assert judge.score("output", ex()) == pytest.approx(0.8)

    def test_clamps_out_of_range_grades(self) -> None:
        assert LLMJudge(ScriptedModel("15"), rubric="r").score("o", ex()) == 1.0
        assert LLMJudge(ScriptedModel("-3"), rubric="r").score("o", ex()) == 0.0

    def test_example_rubric_overrides_judge_rubric(self) -> None:
        model = ScriptedModel("5")
        judge = LLMJudge(model, rubric="default")
        assert judge.score("o", ex(rubric="specific")) == 0.5
        assert model.call_count == 1

    def test_missing_rubric_raises(self) -> None:
        with pytest.raises(DegenerateEvalError, match="no rubric"):
            LLMJudge(ScriptedModel("5")).score("o", ex())

    def test_unparseable_reply_raises(self) -> None:
        judge = LLMJudge(ScriptedModel("looks fine to me"), rubric="r")
        with pytest.raises(DegenerateEvalError, match="no numeric score"):
            judge.score("o", ex())

    def test_flagged_model_graded(self) -> None:
        assert LLMJudge(ScriptedModel("5"), rubric="r").model_graded is True
