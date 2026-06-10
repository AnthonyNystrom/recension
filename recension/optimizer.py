"""The propose/test/accept loop.

:class:`ReflectiveOptimizer` holds the model fixed and optimizes the text
artifact against held-out evidence: diagnose failures on the train split,
propose distinct candidate edits, score them on the validation split, and
accept only a candidate that beats the incumbent by ``min_improvement`` and
survives the leakage checks. Every decision lands in the returned
:class:`~recension.record.RunRecord`.
"""

from __future__ import annotations

import time
import zlib
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from .artifact import Provenance, RejectedCandidate, TextArtifact, _diff_texts
from .budget import Budget
from .evalset import EvalSet, Example
from .exceptions import BudgetExceeded, LeakageDetected
from .leakage import check_candidate
from .models.base import Message, Model
from .objective import Objective
from .proposer import FailureCase, diagnose, propose
from .record import CandidateRecord, RoundRecord, RunRecord

__all__ = ["ReflectiveOptimizer"]

Renderer = Callable[[str, Example], list[Message]]
ProgressCallback = Callable[[str], None]


def _default_render(artifact_text: str, example: Example) -> list[Message]:
    # DESIGN NOTE: the PRD does not fix how an artifact is applied to an
    # example. Default convention: the artifact is the system prompt, the
    # example input is the user message. Callers with other shapes (context
    # templates, instruction files spliced into larger prompts) pass `render`.
    return [
        {"role": "system", "content": artifact_text},
        {"role": "user", "content": example.input},
    ]


class _CallCounter:
    """A single mutable model-call tally shared across guarded models."""

    def __init__(self) -> None:
        self.count = 0


class _GuardedModel:
    """Wraps a model to enforce ``Budget.max_model_calls`` against a shared tally.

    Every guarded ``complete`` increments the shared :class:`_CallCounter` and
    gates on it. Because the optimizer wraps *both* its task model and (for a
    model-graded objective) the judge model in guards that share one counter,
    the ceiling is enforced over the combined call volume, so judge calls cannot
    slip past it. The counter, not any underlying model's own ``call_count``,
    is the budget's source of truth.
    """

    def __init__(self, model: Model, max_model_calls: int | None, counter: _CallCounter) -> None:
        self._model = model
        self._max = max_model_calls
        self._counter = counter

    @property
    def call_count(self) -> int:
        return self._counter.count

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        if self._max is not None and self._counter.count >= self._max:
            raise BudgetExceeded(f"max_model_calls={self._max} reached")
        self._counter.count += 1
        return self._model.complete(
            messages, max_tokens=max_tokens, temperature=temperature, seed=seed
        )


@dataclass(frozen=True)
class _TrainEval:
    """Cached train scoring for one artifact text, threaded between rounds.

    Lets a round skip re-scoring an incumbent whose text is unchanged from the
    previous round (a deterministic, otherwise-redundant model spend).
    """

    text: str
    score: float
    scores: list[float]
    outputs: list[str]


class ReflectiveOptimizer:
    """Optimizes a :class:`TextArtifact` against a frozen model.

    Args:
        artifact: The text under optimization. Mutated in place: accepted
            candidates are committed to it with full provenance.
        evalset: Held-out examples. Failures are diagnosed on ``train``;
            acceptance is decided only on ``validation``.
        objective: Scoring function; higher is better.
        model: The frozen model, used for task execution, diagnosis, and
            proposals. The model is never changed by the optimizer; only
            the text is.
        budget: Update-time compute limits. Defaults to ``Budget()``.
        seed: Optional seed forwarded (derived per call) to the model for
            reproducible runs against :class:`~recension.models.MockModel`.
        min_improvement: A candidate must beat the incumbent's validation
            score by more than this to be accepted.
        strict_leakage: When True, a winning candidate that trips a leakage
            heuristic raises :class:`LeakageDetected` (with the partial record
            attached) instead of being accepted with flags.
        stop_on_no_improvement: Stop after the first round whose best
            candidate fails to beat the incumbent, instead of spending the
            remaining rounds.
        task_max_tokens: ``max_tokens`` for task (example-scoring) calls.
        render: Maps ``(artifact_text, example)`` to the messages sent to the
            model. See ``_default_render`` for the default convention.
        on_progress: Optional callback receiving human-readable progress lines.

    Raises:
        BudgetExceeded: When ``budget.max_model_calls`` is hit (counting task,
            diagnosis, proposal, and judge calls). The partial run record,
            including the round in progress, is attached as ``.record``.
        LeakageDetected: In strict mode, when the winning candidate trips a
            leakage heuristic. Partial record attached as ``.record``.
    """

    def __init__(
        self,
        artifact: TextArtifact,
        evalset: EvalSet,
        objective: Objective,
        model: Model,
        budget: Budget | None = None,
        *,
        seed: int | None = None,
        min_improvement: float = 1e-6,
        strict_leakage: bool = False,
        stop_on_no_improvement: bool = True,
        task_max_tokens: int = 1024,
        render: Renderer | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.artifact = artifact
        self.evalset = evalset
        self.objective = objective
        self.budget = budget or Budget()
        self.seed = seed
        self.min_improvement = min_improvement
        self.strict_leakage = strict_leakage
        self.stop_on_no_improvement = stop_on_no_improvement
        self.task_max_tokens = task_max_tokens
        self._render: Renderer = render or _default_render
        self._on_progress = on_progress
        self._counter = _CallCounter()
        self._model = _GuardedModel(model, self.budget.max_model_calls, self._counter)

    # -- public -------------------------------------------------------------

    def run(self) -> RunRecord:
        """Execute the optimization loop and return the complete audit record."""
        started_at = datetime.now(UTC).isoformat()
        rounds: list[RoundRecord] = []
        baseline_version_id = self.artifact.current().version_id
        baseline_score = float("nan")
        incumbent_score = float("nan")
        restore = self._gate_objective_model()
        try:
            baseline_score, _, _ = self._score_set(self.artifact.text, self.evalset.validation)
            self._progress(f"baseline validation score: {baseline_score:.4f}")
            incumbent_score = baseline_score
            stopped_reason = "completed"
            train_cache: _TrainEval | None = None
            for round_index in range(1, self.budget.rounds + 1):
                round_record, incumbent_score, train_cache = self._run_round(
                    round_index, incumbent_score, train_cache
                )
                rounds.append(round_record)
                if round_record.accepted_version_id is None and self.stop_on_no_improvement:
                    stopped_reason = "no_improvement"
                    break
            return self._build_record(
                started_at, baseline_version_id, baseline_score, rounds,
                incumbent_score, stopped_reason,
            )
        except _BudgetInRound as exc:
            rounds.append(exc.round_record)
            partial = self._build_record(
                started_at, baseline_version_id, baseline_score, rounds,
                incumbent_score, "budget_exceeded",
            )
            raise BudgetExceeded(str(exc.original), record=partial) from exc.original
        except BudgetExceeded as exc:
            # Hit during baseline scoring, before any round started.
            partial = self._build_record(
                started_at, baseline_version_id, baseline_score, rounds,
                incumbent_score, "budget_exceeded",
            )
            raise BudgetExceeded(str(exc), record=partial) from exc
        except _StrictLeakage as exc:
            rounds.append(exc.round_record)
            partial = self._build_record(
                started_at, baseline_version_id, baseline_score, rounds,
                incumbent_score, "leakage_detected",
            )
            raise LeakageDetected(
                f"candidate {exc.candidate_id} tripped leakage heuristics in strict mode: "
                + "; ".join(exc.flags),
                record=partial,
            ) from None
        finally:
            if restore is not None:
                restore()

    # -- internals ----------------------------------------------------------

    def _gate_objective_model(self) -> Callable[[], None] | None:
        """Route a model-graded objective's judge calls through the budget gate.

        # DESIGN NOTE: a model-graded objective (LLMJudge) holds its own model,
        # so its calls would otherwise bypass the optimizer's ceiling. We swap
        # that model for a guard sharing this run's counter for the duration of
        # run(), restoring the caller's original object afterwards (no permanent
        # mutation, and no assumption that task and judge share one instance).
        """
        objective: Any = self.objective
        if not getattr(objective, "model_graded", False) or not hasattr(objective, "model"):
            return None
        original = objective.model
        objective.model = _GuardedModel(original, self.budget.max_model_calls, self._counter)

        def restore() -> None:
            objective.model = original

        return restore

    def _run_round(
        self,
        round_index: int,
        incumbent_score: float,
        train_cache: _TrainEval | None,
    ) -> tuple[RoundRecord, float, _TrainEval]:
        t0 = time.perf_counter()
        calls_before = self._model.call_count
        incumbent_text = self.artifact.text
        incumbent_version_id = self.artifact.current().version_id

        # Progressive state, captured into a partial record if the budget runs
        # out partway through the round (so the audit trail keeps what was done).
        train_score = float("nan")
        train_scores: list[float] = []
        train_outputs: list[str] = []
        failures: list[FailureCase] = []
        diagnosis = ""
        scored: list[tuple[str, str, float, list[str]]] = []  # (id, text, score, flags)
        accepted_train_score: float | None = None

        try:
            if train_cache is not None and train_cache.text == incumbent_text:
                self._progress(f"round {round_index}: reusing cached incumbent train score")
                train_score = train_cache.score
                train_scores = train_cache.scores
                train_outputs = train_cache.outputs
            else:
                self._progress(f"round {round_index}: scoring incumbent on train")
                train_score, train_scores, train_outputs = self._score_set(
                    incumbent_text, self.evalset.train
                )

            failures = self._collect_failures(train_scores, train_outputs)
            diagnosis = diagnose(
                self._model,
                incumbent_text,
                failures,
                seed=self._seed_for(f"diagnose:{round_index}"),
            )
            self._progress(f"round {round_index}: diagnosis: {diagnosis}")

            candidate_texts = propose(
                self._model,
                incumbent_text,
                diagnosis,
                self.budget.candidates_per_round,
                seed=self._seed_for(f"propose:{round_index}"),
            )
            if len(candidate_texts) < self.budget.candidates_per_round:
                self._progress(
                    f"round {round_index}: only {len(candidate_texts)} distinct candidate(s) "
                    f"of {self.budget.candidates_per_round} requested"
                )

            for i, text in enumerate(candidate_texts, 1):
                candidate_id = f"r{round_index}-c{i}"
                score, _, _ = self._score_set(text, self.evalset.validation)
                flags = [str(f) for f in check_candidate(text, incumbent_text, self.evalset)]
                scored.append((candidate_id, text, score, flags))
                self._progress(
                    f"round {round_index}: candidate {candidate_id} scored {score:.4f}"
                )

            accepted_id: str | None = None
            accepted_version_id: str | None = None
            new_incumbent_score = incumbent_score
            next_cache = _TrainEval(incumbent_text, train_score, train_scores, train_outputs)

            if scored:
                best_id, best_text, best_score, _ = max(scored, key=lambda s: s[2])
                if best_score > incumbent_score + self.min_improvement:
                    cand_train_score, cand_train_scores, cand_train_outputs = self._score_set(
                        best_text, self.evalset.train
                    )
                    accepted_train_score = cand_train_score
                    full_flags = [
                        str(f)
                        for f in check_candidate(
                            best_text,
                            incumbent_text,
                            self.evalset,
                            train_gain=cand_train_score - train_score,
                            validation_gain=best_score - incumbent_score,
                        )
                    ]
                    scored = [
                        (cid, text, score, full_flags if cid == best_id else flags)
                        for cid, text, score, flags in scored
                    ]
                    if full_flags and self.strict_leakage:
                        round_record = self._round_record(
                            round_index, incumbent_version_id, incumbent_score, train_score,
                            failures, diagnosis, scored, None, None, None,
                            calls_before, t0,
                        )
                        raise _StrictLeakage(round_record, best_id, full_flags)
                    if full_flags:
                        self._progress(
                            f"round {round_index}: accepting {best_id} WITH leakage flags: "
                            + "; ".join(full_flags)
                        )
                    accepted_id = best_id
                    version = self.artifact.commit(
                        best_text,
                        Provenance(
                            diagnosis=diagnosis,
                            failure_example_ids=tuple(f.example.id for f in failures),
                            incumbent_score=incumbent_score,
                            candidate_score=best_score,
                            rejected_candidates=tuple(
                                RejectedCandidate(cid, text, score, tuple(flags))
                                for cid, text, score, flags in scored
                                if cid != best_id
                            ),
                        ),
                    )
                    accepted_version_id = version.version_id
                    new_incumbent_score = best_score
                    next_cache = _TrainEval(
                        best_text, cand_train_score, cand_train_scores, cand_train_outputs
                    )
                    self._progress(
                        f"round {round_index}: accepted {best_id} as version "
                        f"{accepted_version_id}"
                    )
                else:
                    self._progress(f"round {round_index}: no candidate beat the incumbent")
        except BudgetExceeded as exc:
            partial = self._round_record(
                round_index, incumbent_version_id, incumbent_score, train_score,
                failures, diagnosis, scored, None, None, None,
                calls_before, t0,
            )
            raise _BudgetInRound(partial, exc) from exc

        round_record = self._round_record(
            round_index, incumbent_version_id, incumbent_score, train_score,
            failures, diagnosis, scored, accepted_id, accepted_version_id,
            accepted_train_score, calls_before, t0,
        )
        return round_record, new_incumbent_score, next_cache

    def _round_record(
        self,
        round_index: int,
        incumbent_version_id: str,
        incumbent_score: float,
        train_score: float,
        failures: list[FailureCase],
        diagnosis: str,
        scored: list[tuple[str, str, float, list[str]]],
        accepted_id: str | None,
        accepted_version_id: str | None,
        accepted_train_score: float | None,
        calls_before: int,
        t0: float,
    ) -> RoundRecord:
        incumbent_text_at_start = self.artifact.get(incumbent_version_id).text
        return RoundRecord(
            round_index=round_index,
            incumbent_version_id=incumbent_version_id,
            incumbent_validation_score=incumbent_score,
            train_score=train_score,
            failure_example_ids=tuple(f.example.id for f in failures),
            diagnosis=diagnosis,
            candidates=tuple(
                CandidateRecord(
                    candidate_id=cid,
                    text=text,
                    validation_score=score,
                    diff=_diff_texts(incumbent_text_at_start, text, incumbent_version_id, cid),
                    leakage_flags=tuple(flags),
                    accepted=cid == accepted_id,
                    train_score=accepted_train_score if cid == accepted_id else None,
                )
                for cid, text, score, flags in scored
            ),
            accepted_version_id=accepted_version_id,
            model_calls_used=self._model.call_count - calls_before,
            elapsed_seconds=time.perf_counter() - t0,
        )

    def _build_record(
        self,
        started_at: str,
        baseline_version_id: str,
        baseline_score: float,
        rounds: list[RoundRecord],
        final_score: float,
        stopped_reason: str,
    ) -> RunRecord:
        return RunRecord(
            artifact=self.artifact.to_dict(),
            objective_name=self.objective.name,
            model_graded=self.objective.model_graded,
            seed=self.seed,
            budget=self.budget.to_dict(),
            baseline_version_id=baseline_version_id,
            baseline_score=baseline_score,
            rounds=list(rounds),
            final_version_id=self.artifact.current().version_id,
            final_score=final_score,
            total_model_calls=self._model.call_count,
            stopped_reason=stopped_reason,
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
        )

    def _score_set(
        self, artifact_text: str, examples: Sequence[Example]
    ) -> tuple[float, list[float], list[str]]:
        """Score ``artifact_text`` on ``examples``; returns (aggregate, scores, outputs)."""
        scores: list[float] = []
        outputs: list[str] = []
        for example in examples:
            output = self._model.complete(
                self._render(artifact_text, example),
                max_tokens=self.task_max_tokens,
                temperature=0.0,
                seed=self._seed_for(f"task:{example.id}"),
            )
            outputs.append(output)
            scores.append(self.objective.score(output, example))
        return self.objective.aggregate(scores), scores, outputs

    def _collect_failures(
        self, train_scores: list[float], train_outputs: list[str]
    ) -> list[FailureCase]:
        indexed = sorted(enumerate(train_scores), key=lambda pair: (pair[1], pair[0]))
        worst = indexed[: self.budget.diagnosis_depth]
        return [
            FailureCase(
                example=self.evalset.train[i], output=train_outputs[i], score=score
            )
            for i, score in worst
        ]

    def _seed_for(self, tag: str) -> int | None:
        if self.seed is None:
            return None
        return zlib.crc32(f"{self.seed}:{tag}".encode())

    def _progress(self, message: str) -> None:
        if self._on_progress is not None:
            self._on_progress(message)


class _StrictLeakage(Exception):
    """Internal carrier so run() can attach the full partial record."""

    def __init__(self, round_record: RoundRecord, candidate_id: str, flags: list[str]) -> None:
        super().__init__()
        self.round_record = round_record
        self.candidate_id = candidate_id
        self.flags = flags


class _BudgetInRound(Exception):
    """Internal carrier: a budget overrun mid-round, with the partial round."""

    def __init__(self, round_record: RoundRecord, original: BudgetExceeded) -> None:
        super().__init__()
        self.round_record = round_record
        self.original = original
