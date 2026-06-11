"""The audit record: a complete, serializable history of an optimization run.

A :class:`RunRecord` must be complete enough that a reviewer who did not run
the optimization can reconstruct every decision: the baseline, every round's
diagnosis, every candidate (accepted and rejected) with its scores and leakage
flags, the diffs, the model-call counts, and why the run stopped. The full
artifact (with version history) is embedded so the record stands alone.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .artifact import TextArtifact


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/inf) with None.

    A partial record from a run that was cut short before a score could be
    computed carries ``float('nan')``; emitting it raw would produce invalid,
    non-interoperable JSON. We serialize such an uncomputed score as ``null``;
    :func:`_restore_score` maps it back on load.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    return obj


def _restore_score(value: Any) -> float:
    """Inverse of the NaN->null mapping for a required score field."""
    return float("nan") if value is None else value

__all__ = [
    "CandidateRecord",
    "GuardScore",
    "RoundRecord",
    "RunRecord",
    "SignificanceRecord",
    "SliceScore",
]


@dataclass(frozen=True)
class GuardScore:
    """A guard objective's incumbent-vs-candidate score for the best candidate.

    Recorded when the optimizer runs with ``guards=[...]``, so a reviewer sees
    why a candidate was held back (or that it cleared the guards).
    """

    name: str
    incumbent_score: float
    candidate_score: float

    @property
    def regressed(self) -> bool:
        """True when the candidate scores worse than the incumbent on this guard."""
        return self.candidate_score < self.incumbent_score


@dataclass(frozen=True)
class SliceScore:
    """Baseline vs final score for one subgroup of the validation set.

    Recorded per distinct value of the optimizer's ``slice_by`` metadata key, so
    a run that improves overall but regresses a segment is visible rather than
    averaged away.
    """

    slice: str
    n: int
    baseline_score: float
    final_score: float

    @property
    def regressed(self) -> bool:
        """True when this slice scored worse at the end than at the start."""
        return self.final_score < self.baseline_score


@dataclass(frozen=True)
class SignificanceRecord:
    """The significance test applied to a candidate's validation gain.

    Recorded for the best candidate of a round when the optimizer runs with
    ``accept_significant=True``, so a reviewer can see not just the score delta
    but whether it cleared the confidence bar.

    Attributes:
        mean_difference: Mean per-example validation gain over the incumbent.
        ci_low: Lower bound of the bootstrap confidence interval on the gain.
        ci_high: Upper bound of that interval.
        alpha: Significance level (the interval is ``1 - alpha``).
        significant: True when the interval excludes 0 (gain significantly > 0).
    """

    mean_difference: float
    ci_low: float
    ci_high: float
    alpha: float
    significant: bool


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
        significance: The significance test on this candidate's validation
            gain, populated for the round's best candidate when the run used
            ``accept_significant``. ``None`` otherwise.
    """

    candidate_id: str
    text: str
    validation_score: float | None
    diff: str
    leakage_flags: tuple[str, ...] = ()
    accepted: bool = False
    train_score: float | None = None
    significance: SignificanceRecord | None = None
    guard_scores: tuple[GuardScore, ...] = ()


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
    input_tokens: int = 0
    output_tokens: int = 0


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
        final_test_score: Score of the final incumbent on the locked test
            split, computed exactly once; ``None`` if no test split was given.
        test_validation_gap: ``final_score - final_test_score`` when a test
            split exists; a large positive gap suggests the validation score
            is optimistic (overfit to validation across rounds). ``None``
            without a test split.
        validation_overfit: True when ``test_validation_gap`` exceeds the
            optimizer's ``overfit_gap`` threshold; surfaced, not hidden.
        total_input_tokens: Total input tokens reported by the model across the
            run (0 if the model does not report usage).
        total_output_tokens: Total output tokens reported across the run.
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
    final_test_score: float | None = None
    test_validation_gap: float | None = None
    validation_overfit: bool = False
    slice_scores: list[SliceScore] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
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
                incumbent_validation_score=_restore_score(r["incumbent_validation_score"]),
                train_score=_restore_score(r["train_score"]),
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
                        significance=_significance_from_dict(c.get("significance")),
                        guard_scores=tuple(
                            GuardScore(
                                name=g["name"],
                                incumbent_score=g["incumbent_score"],
                                candidate_score=g["candidate_score"],
                            )
                            for g in c.get("guard_scores", [])
                        ),
                    )
                    for c in r["candidates"]
                ),
                accepted_version_id=r["accepted_version_id"],
                model_calls_used=r["model_calls_used"],
                elapsed_seconds=r["elapsed_seconds"],
                input_tokens=r.get("input_tokens", 0),
                output_tokens=r.get("output_tokens", 0),
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
            baseline_score=_restore_score(data["baseline_score"]),
            rounds=rounds,
            final_version_id=data["final_version_id"],
            final_score=_restore_score(data["final_score"]),
            final_test_score=data.get("final_test_score"),
            test_validation_gap=data.get("test_validation_gap"),
            validation_overfit=data.get("validation_overfit", False),
            slice_scores=[
                SliceScore(
                    slice=s["slice"],
                    n=s["n"],
                    baseline_score=s["baseline_score"],
                    final_score=s["final_score"],
                )
                for s in data.get("slice_scores", [])
            ],
            total_input_tokens=data.get("total_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_model_calls=data["total_model_calls"],
            stopped_reason=data["stopped_reason"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize to JSON (valid even for partial records with uncomputed scores)."""
        return json.dumps(_json_safe(self.to_dict()), indent=indent, ensure_ascii=False)

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

    # -- integrity ----------------------------------------------------------

    def fingerprint(self) -> str:
        """Deterministic SHA-256 over the canonical record JSON.

        Two records with identical content produce the same fingerprint. Store
        it (or a signature of it) somewhere trusted to detect later tampering
        with any field; the artifact lineage is additionally self-verifiable
        via :meth:`verify`.
        """
        canonical = json.dumps(_json_safe(self.to_dict()), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def verify(self) -> list[str]:
        """Integrity problems with the embedded artifact's version chain.

        Empty when intact. Because version ids are content-addressed, this
        catches tampering with a version's text or id without needing any
        external reference. (Tampering with non-versioned fields such as a
        recorded score is caught instead by comparing :meth:`fingerprint` or a
        signature against a trusted copy.)
        """
        return self.restored_artifact().verify()

    def sign(self, key: str) -> str:
        """HMAC-SHA256 of the fingerprint with ``key`` (hex), for signed records."""
        digest = hmac.new(
            key.encode("utf-8"), self.fingerprint().encode("utf-8"), hashlib.sha256
        )
        return digest.hexdigest()

    def verify_signature(self, key: str, signature: str) -> bool:
        """Constant-time check that ``signature`` matches :meth:`sign` for ``key``."""
        return hmac.compare_digest(self.sign(key), signature)

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
                sig = ""
                if c.significance is not None:
                    s = c.significance
                    verdict = "significant" if s.significant else "NOT significant"
                    sig = (
                        f"  [{verdict}: gain {s.mean_difference:+.4f}, "
                        f"{int((1 - s.alpha) * 100)}% CI [{s.ci_low:+.4f}, {s.ci_high:+.4f}]]"
                    )
                guards = ""
                if c.guard_scores:
                    parts = [
                        f"{g.name} {g.incumbent_score:.4f}->{g.candidate_score:.4f}"
                        + ("!" if g.regressed else "")
                        for g in c.guard_scores
                    ]
                    guards = "  guards: " + ", ".join(parts)
                lines.append(
                    f"  candidate {c.candidate_id}: {score}  [{status}]{flags}{sig}{guards}"
                )
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
        if self.final_test_score is not None:
            gap = self.test_validation_gap if self.test_validation_gap is not None else 0.0
            warn = (
                "  [WARNING: possible overfitting to validation]"
                if self.validation_overfit
                else ""
            )
            lines.append(
                f"test (locked, scored once): {self.final_test_score:.4f}  "
                f"validation/test gap {gap:.4f}{warn}"
            )
        if self.slice_scores:
            lines.append("slices:")
            for sl in self.slice_scores:
                mark = "  [REGRESSED]" if sl.regressed else ""
                lines.append(
                    f"  {sl.slice} (n={sl.n}): "
                    f"{sl.baseline_score:.4f} -> {sl.final_score:.4f}{mark}"
                )
        lines.append(f"model calls: {self.total_model_calls}")
        if self.total_input_tokens or self.total_output_tokens:
            lines.append(
                f"tokens: {self.total_input_tokens} in / {self.total_output_tokens} out"
            )
        lines.append(f"stopped: {self.stopped_reason}")
        return "\n".join(lines)


def _significance_from_dict(data: dict[str, Any] | None) -> SignificanceRecord | None:
    if data is None:
        return None
    return SignificanceRecord(
        mean_difference=data["mean_difference"],
        ci_low=data["ci_low"],
        ci_high=data["ci_high"],
        alpha=data["alpha"],
        significant=data["significant"],
    )
