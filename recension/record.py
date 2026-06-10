"""The audit record: a complete, serializable history of an optimization run.

A :class:`RunRecord` must be complete enough that a reviewer who did not run
the optimization can reconstruct every decision: the baseline, every round's
diagnosis, every candidate (accepted and rejected) with its scores and leakage
flags, the diffs, the model-call counts, and why the run stopped. The full
artifact (with version history) is embedded so the record stands alone.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .artifact import TextArtifact

__all__ = ["CandidateRecord", "RoundRecord", "RunRecord"]


@dataclass(frozen=True)
class CandidateRecord:
    """One candidate edit considered in a round.

    Attributes:
        candidate_id: Stable id within the run (e.g. ``"r1-c2"``).
        text: The full candidate artifact text.
        validation_score: Aggregate held-out score, or ``None`` if scoring
            was cut short (e.g. budget exhausted).
        diff: Unified diff against the incumbent at proposal time.
        leakage_flags: Human-readable descriptions of tripped heuristics.
        accepted: Whether this candidate became the new incumbent.
        train_score: Aggregate train score of this candidate, populated only
            for the accepted candidate (the only one re-scored on train, to
            check the implausible-gain heuristic). ``None`` otherwise. With
            ``RoundRecord.train_score`` (the incumbent's train score) this
            lets a reviewer reconstruct the train-vs-validation gain that
            drove the leakage decision.
    """

    candidate_id: str
    text: str
    validation_score: float | None
    diff: str
    leakage_flags: tuple[str, ...] = ()
    accepted: bool = False
    train_score: float | None = None


@dataclass(frozen=True)
class RoundRecord:
    """Everything that happened in one optimization round."""

    round_index: int
    incumbent_version_id: str
    incumbent_validation_score: float
    train_score: float
    failure_example_ids: tuple[str, ...]
    diagnosis: str
    candidates: tuple[CandidateRecord, ...]
    accepted_version_id: str | None
    model_calls_used: int
    elapsed_seconds: float


@dataclass
class RunRecord:
    """Complete, serializable history of one ``ReflectiveOptimizer.run()``.

    Attributes:
        artifact: Full snapshot (``TextArtifact.to_dict()``) of the artifact
            after the run, version history and provenance included.
        objective_name: The objective used for all scoring.
        model_graded: True if the objective itself calls a model
            (:class:`~recension.objective.LLMJudge`); flagged so reviewers
            know the acceptance metric is not reference-based.
        seed: The seed the optimizer was constructed with, if any.
        budget: ``Budget.to_dict()`` snapshot.
        baseline_version_id: Incumbent version at the start of the run.
        baseline_score: Held-out validation score of the baseline.
        rounds: One :class:`RoundRecord` per executed round.
        final_version_id: Incumbent version at the end of the run.
        final_score: Held-out validation score of the final incumbent.
        total_model_calls: All model calls spent, including judge calls.
        stopped_reason: Why the run ended (``"completed"``,
            ``"no_improvement"``, ``"budget_exceeded"``).
        started_at: ISO 8601 timestamp.
        finished_at: ISO 8601 timestamp.
    """

    artifact: dict[str, Any]
    objective_name: str
    model_graded: bool
    seed: int | None
    budget: dict[str, Any]
    baseline_version_id: str
    baseline_score: float
    rounds: list[RoundRecord] = field(default_factory=list)
    final_version_id: str = ""
    final_score: float = 0.0
    total_model_calls: int = 0
    stopped_reason: str = ""
    started_at: str = ""
    finished_at: str = ""

    # -- serialization ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form of the whole record."""
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Inverse of :meth:`to_dict`."""
        rounds = [
            RoundRecord(
                round_index=r["round_index"],
                incumbent_version_id=r["incumbent_version_id"],
                incumbent_validation_score=r["incumbent_validation_score"],
                train_score=r["train_score"],
                failure_example_ids=tuple(r["failure_example_ids"]),
                diagnosis=r["diagnosis"],
                candidates=tuple(
                    CandidateRecord(
                        candidate_id=c["candidate_id"],
                        text=c["text"],
                        validation_score=c["validation_score"],
                        diff=c["diff"],
                        leakage_flags=tuple(c["leakage_flags"]),
                        accepted=c["accepted"],
                        train_score=c.get("train_score"),
                    )
                    for c in r["candidates"]
                ),
                accepted_version_id=r["accepted_version_id"],
                model_calls_used=r["model_calls_used"],
                elapsed_seconds=r["elapsed_seconds"],
            )
            for r in data["rounds"]
        ]
        return cls(
            artifact=data["artifact"],
            objective_name=data["objective_name"],
            model_graded=data["model_graded"],
            seed=data["seed"],
            budget=data["budget"],
            baseline_version_id=data["baseline_version_id"],
            baseline_score=data["baseline_score"],
            rounds=rounds,
            final_version_id=data["final_version_id"],
            final_score=data["final_score"],
            total_model_calls=data["total_model_calls"],
            stopped_reason=data["stopped_reason"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> RunRecord:
        """Inverse of :meth:`to_json`."""
        return cls.from_dict(json.loads(payload))

    def save(self, path: str | Path) -> None:
        """Write the record to a JSON file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> RunRecord:
        """Read a record from a JSON file."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))

    # -- reporting ----------------------------------------------------------

    def restored_artifact(self) -> TextArtifact:
        """Rehydrate the embedded artifact (for diffs and inspection)."""
        return TextArtifact.from_dict(self.artifact)

    def summary(self) -> str:
        """Human-readable account of the run: baseline, rounds, diffs, scores."""
        lines = [
            f"artifact: {self.artifact.get('name', 'artifact')}",
            f"objective: {self.objective_name}"
            + (" (model-graded)" if self.model_graded else ""),
            f"baseline: version {self.baseline_version_id}  "
            f"validation score {self.baseline_score:.4f}",
        ]
        for r in self.rounds:
            lines.append("")
            lines.append(
                f"round {r.round_index}: train score {r.train_score:.4f}, "
                f"failures analyzed: {', '.join(r.failure_example_ids) or 'none'}"
            )
            lines.append(f"  diagnosis: {r.diagnosis}")
            for c in r.candidates:
                score = "n/a" if c.validation_score is None else f"{c.validation_score:.4f}"
                status = "ACCEPTED" if c.accepted else "rejected"
                flags = f"  flags: {', '.join(c.leakage_flags)}" if c.leakage_flags else ""
                lines.append(f"  candidate {c.candidate_id}: {score}  [{status}]{flags}")
            if r.accepted_version_id:
                lines.append(f"  new incumbent: version {r.accepted_version_id}")
                accepted = next((c for c in r.candidates if c.accepted), None)
                if accepted is None:
                    # Defensive: a complete record always has the accepted
                    # candidate, but a hand-edited or truncated one might not.
                    lines.append("  (accepted candidate not found in record)")
                elif accepted.diff:
                    lines.append("  diff:")
                    lines.extend(f"    {line}" for line in accepted.diff.splitlines())
        lines.append("")
        lines.append(
            f"final: version {self.final_version_id}  "
            f"validation score {self.final_score:.4f}  "
            f"({self.baseline_score:.4f} -> {self.final_score:.4f})"
        )
        lines.append(f"model calls: {self.total_model_calls}")
        lines.append(f"stopped: {self.stopped_reason}")
        return "\n".join(lines)
