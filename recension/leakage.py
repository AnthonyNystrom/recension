"""Leakage and overfitting heuristics for candidate artifacts.

These checks are *heuristics*, not proofs. They catch the two cheapest ways an
edit can game a held-out score:

1. **Verbatim validation spans**: the candidate text embeds a long span
   copied from a validation example (its input, expected output, or rubric).
   That is memorization of the held-out set, not generalization.
2. **Implausible gain**: the candidate's validation gain is large while its
   train gain is flat or negative. Honest improvements usually move both;
   a validation-only jump suggests the edit exploits validation-specific cues.

A tripped heuristic produces a :class:`LeakageFlag`. By default flags are
*surfaced* in the run record, not silently enforced; the optimizer's strict
mode turns them into :class:`~recension.exceptions.LeakageDetected`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .evalset import EvalSet

__all__ = ["LeakageFlag", "check_candidate"]

#: Minimum length (in characters) of a copied span worth flagging.
DEFAULT_MIN_SPAN_LENGTH = 24

#: Validation gain at or above which the implausible-gain heuristic can fire.
DEFAULT_MIN_VALIDATION_GAIN = 0.15

#: Fire when validation gain exceeds this multiple of a positive train gain.
DEFAULT_GAIN_RATIO = 4.0


@dataclass(frozen=True)
class LeakageFlag:
    """One tripped heuristic, with enough detail to review it."""

    kind: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


def check_candidate(
    candidate_text: str,
    incumbent_text: str,
    evalset: EvalSet,
    *,
    train_gain: float | None = None,
    validation_gain: float | None = None,
    min_span_length: int = DEFAULT_MIN_SPAN_LENGTH,
    min_validation_gain: float = DEFAULT_MIN_VALIDATION_GAIN,
    gain_ratio: float = DEFAULT_GAIN_RATIO,
) -> list[LeakageFlag]:
    """Run all leakage heuristics against one candidate.

    Args:
        candidate_text: The proposed artifact text.
        incumbent_text: The current artifact text. Spans already present in
            the incumbent are not re-flagged, only *newly introduced*
            validation content counts.
        evalset: Source of the validation examples to scan for.
        train_gain: Candidate train score minus incumbent train score, if
            measured. Both gains are required for the implausible-gain check.
        validation_gain: Candidate validation score minus incumbent
            validation score, if measured.
        min_span_length: Shortest copied span (characters) considered leakage.
        min_validation_gain: Validation gain below which the implausible-gain
            heuristic never fires (small gains are noise, not leakage).
        gain_ratio: Fire when ``validation_gain > gain_ratio * train_gain``
            (for positive train gain).

    Returns:
        All tripped flags, empty if the candidate looks clean.
    """
    flags = _verbatim_span_flags(candidate_text, incumbent_text, evalset, min_span_length)
    if train_gain is not None and validation_gain is not None:
        flag = _implausible_gain_flag(train_gain, validation_gain, min_validation_gain, gain_ratio)
        if flag is not None:
            flags.append(flag)
    return flags


def _verbatim_span_flags(
    candidate_text: str,
    incumbent_text: str,
    evalset: EvalSet,
    min_span_length: int,
) -> list[LeakageFlag]:
    flags: list[LeakageFlag] = []
    for example in evalset.validation:
        sources = [
            ("input", example.input),
            ("expected", example.expected or ""),
            ("rubric", example.rubric or ""),
        ]
        for label, source in sources:
            span = _longest_new_span(source, candidate_text, incumbent_text, min_span_length)
            if span is not None:
                flags.append(
                    LeakageFlag(
                        kind="verbatim_validation_span",
                        detail=(
                            f"candidate embeds {len(span)} chars of validation example "
                            f"{example.id!r} ({label}): {span[:60]!r}"
                        ),
                    )
                )
                break  # one flag per example is enough to review it
    return flags


def _longest_new_span(
    source: str, candidate_text: str, incumbent_text: str, min_length: int
) -> str | None:
    """Longest span of ``source`` present in the candidate but not the incumbent."""
    if len(source) < min_length:
        return None
    best: str | None = None
    # Greedy scan: at each start, extend the longest substring found in the
    # candidate. Eval sets are small, so the quadratic worst case is fine.
    for start in range(len(source) - min_length + 1):
        end = start + min_length
        if source[start:end] not in candidate_text:
            continue
        while end < len(source) and source[start : end + 1] in candidate_text:
            end += 1
        span = source[start:end]
        if span not in incumbent_text and (best is None or len(span) > len(best)):
            best = span
    return best


def _implausible_gain_flag(
    train_gain: float,
    validation_gain: float,
    min_validation_gain: float,
    gain_ratio: float,
) -> LeakageFlag | None:
    if validation_gain < min_validation_gain:
        return None
    suspicious = train_gain <= 0 or validation_gain > gain_ratio * train_gain
    if not suspicious:
        return None
    return LeakageFlag(
        kind="implausible_gain",
        detail=(
            f"validation gain {validation_gain:+.4f} with train gain {train_gain:+.4f} "
            "(possible memorization of validation cues)"
        ),
    )
