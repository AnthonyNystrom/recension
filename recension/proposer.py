"""Candidate generation: diagnose failures, propose distinct revisions.

The proposer turns observed failures into a structured hypothesis
(:func:`diagnose`) and then into genuinely different candidate edits
(:func:`propose`). Distinctness matters: comparing four rewordings of one idea
tests nothing, so near-duplicate candidates are rejected and regenerated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

from .evalset import Example
from .models.base import Message, Model

__all__ = ["FailureCase", "diagnose", "propose"]

#: Candidates whose similarity ratio exceeds this are treated as duplicates.
NEAR_DUPLICATE_RATIO = 0.95

#: Attempts allowed per requested candidate before giving up on distinctness.
ATTEMPTS_PER_CANDIDATE = 3


@dataclass(frozen=True)
class FailureCase:
    """One failed train example: what went in, what came out, how it scored."""

    example: Example
    output: str
    score: float


_DIAGNOSE_SYSTEM = """\
You analyze why a text artifact (a prompt, context template, or instruction \
file) produced failing outputs from a language model. Read the artifact and \
the failed cases, then state a short, specific hypothesis about what in the \
artifact caused the failures. Talk about the artifact's text, not the model. \
Reply with the hypothesis only.\
"""

_PROPOSE_SYSTEM = """\
You revise text artifacts (prompts, context templates, instruction files) to \
fix diagnosed failures. Produce a complete revised version of the artifact, \
not a patch, not commentary. Reply with the revised artifact text wrapped in \
<revised_artifact> tags and nothing else.\
"""

_CANDIDATE_RE = re.compile(r"<revised_artifact>\n?(.*?)\n?</revised_artifact>", re.DOTALL)


def diagnose(
    model: Model,
    artifact_text: str,
    failures: list[FailureCase],
    *,
    max_tokens: int = 1024,
    seed: int | None = None,
) -> str:
    """Ask the model why the artifact produced these failures.

    Returns the model's hypothesis as free text (recorded verbatim in the
    round record).
    """
    cases = []
    for i, case in enumerate(failures, 1):
        expected = (
            f"\nexpected: {case.example.expected}" if case.example.expected is not None else ""
        )
        cases.append(
            f"<case index={i} example_id={case.example.id!r} score={case.score:.4f}>\n"
            f"input: {case.example.input}\n"
            f"output: {case.output}{expected}\n"
            f"</case>"
        )
    messages: list[Message] = [
        {"role": "system", "content": _DIAGNOSE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"<artifact>\n{artifact_text}\n</artifact>\n\n"
                f"Failed cases:\n\n" + "\n\n".join(cases)
            ),
        },
    ]
    return model.complete(messages, max_tokens=max_tokens, temperature=0.0, seed=seed).strip()


def propose(
    model: Model,
    artifact_text: str,
    diagnosis: str,
    n: int,
    *,
    max_tokens: int = 4096,
    seed: int | None = None,
) -> list[str]:
    """Generate up to ``n`` distinct candidate revisions of the artifact.

    Near-duplicates (of the incumbent or of each other, by
    ``difflib.SequenceMatcher`` ratio) are rejected and regenerated, up to
    ``ATTEMPTS_PER_CANDIDATE`` attempts per requested candidate. If the model
    cannot produce ``n`` distinct candidates within that allowance, the
    distinct subset found so far is returned; the optimizer records how many
    candidates each round actually compared, so a shortfall is visible in the
    audit record rather than silently padded.
    """
    candidates: list[str] = []
    attempts = 0
    max_attempts = n * ATTEMPTS_PER_CANDIDATE
    while len(candidates) < n and attempts < max_attempts:
        attempts += 1
        call_seed = None if seed is None else seed + attempts
        reply = model.complete(
            _proposal_messages(artifact_text, diagnosis, len(candidates) + 1, n, candidates),
            max_tokens=max_tokens,
            temperature=0.0,
            seed=call_seed,
        )
        text = extract_candidate(reply)
        if not text.strip():
            continue
        if _is_near_duplicate(text, artifact_text) or any(
            _is_near_duplicate(text, existing) for existing in candidates
        ):
            continue
        candidates.append(text)
    return candidates


def _proposal_messages(
    artifact_text: str, diagnosis: str, index: int, total: int, existing: list[str]
) -> list[Message]:
    distinct_clause = ""
    if existing:
        previous = "\n\n".join(
            f"<previous_candidate index={i}>\n{text}\n</previous_candidate>"
            for i, text in enumerate(existing, 1)
        )
        distinct_clause = (
            "\n\nCandidates proposed so far are below. Your revision must take a "
            "genuinely different approach from all of them: a different hypothesis "
            "about the fix, not a rewording.\n\n" + previous
        )
    return [
        {"role": "system", "content": _PROPOSE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"<artifact>\n{artifact_text}\n</artifact>\n\n"
                f"<diagnosis>\n{diagnosis}\n</diagnosis>\n\n"
                f"Propose revision {index} of {total}.{distinct_clause}"
            ),
        },
    ]


def extract_candidate(reply: str) -> str:
    """Pull the revised artifact out of a proposal reply.

    Prefers the ``<revised_artifact>`` tags the prompt asks for; falls back to
    the whole reply (stripped) when a model ignores the tagging instruction.
    """
    match = _CANDIDATE_RE.search(reply)
    if match:
        return match.group(1)
    return reply.strip()


def _is_near_duplicate(a: str, b: str) -> bool:
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() > NEAR_DUPLICATE_RATIO
