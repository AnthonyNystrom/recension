"""Objectives: how a model output is scored against an example.

An :class:`Objective` maps ``(model_output, example)`` to a float, higher is
better, and aggregates per-example scores into one number (mean by default).
Ships :class:`ExactMatch`, token-level :class:`F1`, and the model-graded
:class:`LLMJudge`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .evalset import Example
from .exceptions import DegenerateEvalError
from .models.base import Message, Model

__all__ = ["ExactMatch", "F1", "LLMJudge", "Objective"]


@runtime_checkable
class Objective(Protocol):
    """Protocol every objective implements.

    Attributes:
        name: Short identifier recorded in run records.
        model_graded: True when scoring itself calls a model (e.g. a judge).
            Model-graded acceptance is flagged in the audit record so a
            reviewer knows the metric is not reference-based.
    """

    name: str
    model_graded: bool

    def score(self, model_output: str, example: Example) -> float:
        """Score one output against one example; higher is better."""
        ...

    def aggregate(self, scores: Sequence[float]) -> float:
        """Combine per-example scores into a single number."""
        ...


def _mean(scores: Sequence[float]) -> float:
    if not scores:
        raise DegenerateEvalError("cannot aggregate an empty list of scores")
    return sum(scores) / len(scores)


def _require_expected(example: Example, objective_name: str) -> str:
    if example.expected is None:
        raise DegenerateEvalError(
            f"example {example.id!r} has no 'expected' value, required by {objective_name}"
        )
    return example.expected


class ExactMatch:
    """1.0 if the output equals the expected value, else 0.0.

    Comparison strips surrounding whitespace and, unless ``case_sensitive``,
    casefolds both sides.
    """

    model_graded = False

    def __init__(self, *, case_sensitive: bool = False) -> None:
        self.name = "exact_match"
        self.case_sensitive = case_sensitive

    def _norm(self, text: str) -> str:
        text = text.strip()
        return text if self.case_sensitive else text.casefold()

    def score(self, model_output: str, example: Example) -> float:
        """1.0 on a normalized exact match with ``example.expected``, else 0.0."""
        expected = _require_expected(example, self.name)
        return 1.0 if self._norm(model_output) == self._norm(expected) else 0.0

    def aggregate(self, scores: Sequence[float]) -> float:
        """Mean of the per-example scores."""
        return _mean(scores)


class F1:
    """Token-level F1 between the output and the expected value.

    Tokens are whitespace-separated, casefolded words, the conventional
    SQuAD-style metric. Returns 1.0 when both sides are empty.
    """

    model_graded = False

    def __init__(self) -> None:
        self.name = "f1"

    def score(self, model_output: str, example: Example) -> float:
        """Token-level F1 against ``example.expected``."""
        expected = _require_expected(example, self.name)
        out_tokens = model_output.casefold().split()
        gold_tokens = expected.casefold().split()
        if not out_tokens and not gold_tokens:
            return 1.0
        if not out_tokens or not gold_tokens:
            return 0.0
        common = Counter(out_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            return 0.0
        precision = overlap / len(out_tokens)
        recall = overlap / len(gold_tokens)
        return 2 * precision * recall / (precision + recall)

    def aggregate(self, scores: Sequence[float]) -> float:
        """Mean of the per-example scores."""
        return _mean(scores)


_JUDGE_PROMPT = """\
You are grading a model output against a rubric. Read the task input, the \
output, and the rubric, then reply with a single integer score from 0 (worst) \
to 10 (best). Reply with the number only.

<task_input>
{input}
</task_input>

<model_output>
{output}
</model_output>

<rubric>
{rubric}
</rubric>
"""

_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


class LLMJudge:
    """Model-graded objective: scores outputs against a rubric, 0..1.

    Intended as the held-out (validation) judge. Runs flagged as
    ``model_graded`` in the audit record, and every judge call counts toward
    the model-call budget. A per-example ``rubric`` overrides the judge-level
    one.

    Raises:
        DegenerateEvalError: If no rubric is available for an example, or the
            judge reply contains no parseable number, since a silent default would
            corrupt the measurement.
    """

    model_graded = True

    def __init__(self, model: Model, rubric: str | None = None, *, max_tokens: int = 16) -> None:
        self.name = "llm_judge"
        self.model = model
        self.rubric = rubric
        self.max_tokens = max_tokens

    def score(self, model_output: str, example: Example) -> float:
        """Ask the judge model for a 0 to 10 grade and normalize it to 0..1."""
        rubric = example.rubric or self.rubric
        if rubric is None:
            raise DegenerateEvalError(
                f"no rubric for example {example.id!r}: pass one to LLMJudge or the example"
            )
        messages: list[Message] = [
            {
                "role": "user",
                "content": _JUDGE_PROMPT.format(
                    input=example.input, output=model_output, rubric=rubric
                ),
            }
        ]
        reply = self.model.complete(messages, max_tokens=self.max_tokens, temperature=0.0)
        match = _NUMBER.search(reply)
        if match is None:
            raise DegenerateEvalError(
                f"judge reply for example {example.id!r} contains no numeric score: {reply!r}"
            )
        grade = float(match.group())
        return max(0.0, min(grade, 10.0)) / 10.0

    def aggregate(self, scores: Sequence[float]) -> float:
        """Mean of the per-example scores."""
        return _mean(scores)
