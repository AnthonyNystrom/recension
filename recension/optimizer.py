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
from .models.base import Message, Model, SupportsUsage, TokenUsage
from .objective import Objective
from .proposer import DefaultProposer, FailureCase, Proposer
from .record import (
    CandidateRecord,
    GuardScore,
    RoundRecord,
    RunRecord,
    SignificanceRecord,
    SliceScore,
)
from .stats import paired_bootstrap

__all__ = ["ReflectiveOptimizer", "score_artifact"]

Renderer = Callable[[str, Example], list[Message]]
ProgressCallback = Callable[[str], None]


def score_artifact(
    artifact_text: str,
    examples: Sequence[Example],
    objective: Objective,
    model: Model,
    *,
    render: Renderer | None = None,
    task_max_tokens: int = 1024,
) -> float:
    """Score one artifact on a set of examples, without running an optimization.

    The aggregate objective score of ``artifact_text`` over ``examples`` against
    the frozen ``model``. Used by ``recension check`` to compare the current
    artifact to a recorded baseline (a prompt regression test).
    """
    render = render or _default_render
    scores = [
        objective.score(
            model.complete(render(artifact_text, example), max_tokens=task_max_tokens),
            example,
        )
        for example in examples
    ]
    return objective.aggregate(scores)


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
    """A single mutable model-call tally and token ledger shared across guards."""

    def __init__(self) -> None:
        self.count = 0
        self.usage = TokenUsage()


class _GuardedModel:
    """Wraps a model to enforce ``Budget.max_model_calls`` against a shared tally.

    Every guarded ``complete`` increments the shared :class:`_CallCounter`,
    gates on it, and adds the underlying model's reported token usage to the
    shared ledger. Because the optimizer wraps *both* its task model and (for a
    model-graded objective) the judge model in guards that share one counter,
    the ceiling and the ledger cover the combined volume, so judge calls cannot
    slip past either. The counter, not any underlying model's own ``call_count``,
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
        reply = self._model.complete(
            messages, max_tokens=max_tokens, temperature=temperature, seed=seed
        )
        if isinstance(self._model, SupportsUsage):
            self._counter.usage = self._counter.usage + self._model.last_usage
        return reply


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


@dataclass(frozen=True)
class _ValEval:
    """Cached incumbent validation results threaded between rounds.

    Holds the per-example primary scores (for the significance gate) and the
    aggregate score of each guard objective (for the no-regression guard gate).
    """

    primary_scores: list[float]
    guard_scores: list[float]


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
        overfit_gap: When the eval set has a ``test`` split, the run flags
            ``validation_overfit`` if the final validation score exceeds the
            test score by more than this gap.
        accept_significant: When True, a candidate is accepted only if its
            validation gain is *statistically significant* (a paired bootstrap
            CI on the per-example gain that excludes 0), not merely larger than
            ``min_improvement``. Off by default, preserving 0.1.0 behavior.
        alpha: Significance level for the bootstrap CI (default 0.05 = 95%).
        bootstrap_resamples: Resamples for the significance bootstrap.
        slice_by: An ``Example.metadata`` key. When set, the record reports
            per-slice baseline-vs-final scores so a run that improves overall
            but regresses a subgroup is visible. ``None`` disables slicing.
        slice_tolerance: A slice is announced as regressed when its score drops
            by more than this (default 0.0).
        guards: Secondary objectives that must not regress. A candidate that
            improves the primary objective but lowers any guard's validation
            score (beyond ``guard_tolerance``) is rejected. Guards are scored
            from the same model outputs, so cheap reference-free guards (e.g.
            :class:`~recension.objective.MaxLength`) add no model calls.
        guard_tolerance: Allowed drop on a guard before it counts as a
            regression (default 0.0).
        proposer: The candidate generator. Defaults to the built-in
            :class:`~recension.proposer.DefaultProposer`; inject a custom
            :class:`~recension.proposer.Proposer` (e.g. one wrapping an external
            optimizer) to supply edits while recension keeps owning measurement,
            leakage, and the audit record.
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
        overfit_gap: float = 0.1,
        accept_significant: bool = False,
        alpha: float = 0.05,
        bootstrap_resamples: int = 2000,
        slice_by: str | None = None,
        slice_tolerance: float = 0.0,
        guards: Sequence[Objective] = (),
        guard_tolerance: float = 0.0,
        proposer: Proposer | None = None,
        task_max_tokens: int = 1024,
        render: Renderer | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> None:
        self.artifact = artifact
        self.evalset = evalset
        self.objective = objective
        self.budget = budget or Budget()
        self.proposer: Proposer = proposer or DefaultProposer()
        self.seed = seed
        self.min_improvement = min_improvement
        self.strict_leakage = strict_leakage
        self.stop_on_no_improvement = stop_on_no_improvement
        self.overfit_gap = overfit_gap
        self.accept_significant = accept_significant
        self.alpha = alpha
        self.bootstrap_resamples = bootstrap_resamples
        self.slice_by = slice_by
        self.slice_tolerance = slice_tolerance
        self.guards = tuple(guards)
        self.guard_tolerance = guard_tolerance
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
            baseline_score, baseline_val_scores, baseline_val_outputs = self._score_set(
                self.artifact.text, self.evalset.validation
            )
            self._progress(f"baseline validation score: {baseline_score:.4f}")
            incumbent_score = baseline_score
            stopped_reason = "completed"
            train_cache: _TrainEval | None = None
            # Incumbent validation results threaded for the significance + guard gates.
            val_cache = _ValEval(baseline_val_scores, self._guard_scores(baseline_val_outputs))
            for round_index in range(1, self.budget.rounds + 1):
                round_record, incumbent_score, train_cache, val_cache = self._run_round(
                    round_index, incumbent_score, train_cache, val_cache
                )
                rounds.append(round_record)
                if round_record.accepted_version_id is None and self.stop_on_no_improvement:
                    stopped_reason = "no_improvement"
                    break
            slice_scores = self._slice_scores(baseline_val_scores, val_cache.primary_scores)
            # The locked test split is a final estimate, not optimization spend.
            # If the budget is exhausted before it can be scored, skip it (and say
            # so) rather than discarding the completed run as a budget failure.
            try:
                final_test_score, gap, overfit = self._score_test(incumbent_score)
            except BudgetExceeded:
                self._progress(
                    "budget exhausted before the locked test split could be scored; skipping it"
                )
                final_test_score, gap, overfit = None, None, False
            return self._build_record(
                started_at, baseline_version_id, baseline_score, rounds,
                incumbent_score, stopped_reason,
                final_test_score=final_test_score,
                test_validation_gap=gap,
                validation_overfit=overfit,
                slice_scores=slice_scores,
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
        val_cache: _ValEval,
    ) -> tuple[RoundRecord, float, _TrainEval, _ValEval]:
        t0 = time.perf_counter()
        calls_before = self._model.call_count
        usage_before = self._counter.usage
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
        val_scores_by_id: dict[str, list[float]] = {}  # per-example validation scores
        guard_scores_by_id: dict[str, list[float]] = {}  # per-candidate guard aggregates
        accepted_train_score: float | None = None
        significance: SignificanceRecord | None = None
        significance_id: str | None = None
        guard_records: tuple[GuardScore, ...] = ()
        guard_id: str | None = None
        next_val_cache = val_cache

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
            diagnosis = self.proposer.diagnose(
                self._model,
                incumbent_text,
                failures,
                seed=self._seed_for(f"diagnose:{round_index}"),
            )
            self._progress(f"round {round_index}: diagnosis: {diagnosis}")

            candidate_texts = self.proposer.propose(
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
                score, val_scores, val_outputs = self._score_set(text, self.evalset.validation)
                val_scores_by_id[candidate_id] = val_scores
                guard_scores_by_id[candidate_id] = self._guard_scores(val_outputs)
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
                beats_threshold = best_score > incumbent_score + self.min_improvement
                accept = beats_threshold

                # Significance gate: a candidate that clears the margin must also
                # show a statistically significant per-example validation gain.
                if beats_threshold and self.accept_significant:
                    boot = paired_bootstrap(
                        val_cache.primary_scores,
                        val_scores_by_id[best_id],
                        alpha=self.alpha,
                        n_resamples=self.bootstrap_resamples,
                        seed=self._seed_for(f"significance:{round_index}:{best_id}"),
                    )
                    significance = SignificanceRecord(
                        mean_difference=boot.mean_difference,
                        ci_low=boot.ci_low,
                        ci_high=boot.ci_high,
                        alpha=boot.alpha,
                        significant=boot.significant,
                    )
                    significance_id = best_id
                    if not boot.significant:
                        accept = False
                        self._progress(
                            f"round {round_index}: {best_id} beat the incumbent by "
                            f"{best_score - incumbent_score:+.4f} but the gain is not "
                            f"significant ({int((1 - boot.alpha) * 100)}% CI "
                            f"[{boot.ci_low:+.4f}, {boot.ci_high:+.4f}]); rejecting"
                        )

                # Guard gate: the best candidate must not regress any guard objective.
                if beats_threshold and self.guards:
                    incumbent_guards = val_cache.guard_scores
                    candidate_guards = guard_scores_by_id[best_id]
                    guard_records = tuple(
                        GuardScore(guard.name, incumbent_guards[gi], candidate_guards[gi])
                        for gi, guard in enumerate(self.guards)
                    )
                    guard_id = best_id
                    regressed = [
                        gr.name
                        for gr in guard_records
                        if gr.candidate_score < gr.incumbent_score - self.guard_tolerance
                    ]
                    if regressed and accept:
                        accept = False
                        self._progress(
                            f"round {round_index}: {best_id} regressed guard(s) "
                            f"{', '.join(regressed)}; rejecting"
                        )

                if accept:
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
                            significance_id, significance, guard_id, guard_records,
                            calls_before, usage_before, t0,
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
                    next_val_cache = _ValEval(
                        val_scores_by_id[best_id], guard_scores_by_id[best_id]
                    )
                    self._progress(
                        f"round {round_index}: accepted {best_id} as version "
                        f"{accepted_version_id}"
                    )
                elif not beats_threshold:
                    self._progress(f"round {round_index}: no candidate beat the incumbent")
        except BudgetExceeded as exc:
            partial = self._round_record(
                round_index, incumbent_version_id, incumbent_score, train_score,
                failures, diagnosis, scored, None, None, None,
                significance_id, significance, guard_id, guard_records,
                calls_before, usage_before, t0,
            )
            raise _BudgetInRound(partial, exc) from exc

        round_record = self._round_record(
            round_index, incumbent_version_id, incumbent_score, train_score,
            failures, diagnosis, scored, accepted_id, accepted_version_id,
            accepted_train_score, significance_id, significance, guard_id, guard_records,
            calls_before, usage_before, t0,
        )
        return round_record, new_incumbent_score, next_cache, next_val_cache

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
        significance_id: str | None,
        significance: SignificanceRecord | None,
        guard_id: str | None,
        guard_records: tuple[GuardScore, ...],
        calls_before: int,
        usage_before: TokenUsage,
        t0: float,
    ) -> RoundRecord:
        incumbent_text_at_start = self.artifact.get(incumbent_version_id).text
        usage = self._counter.usage
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
                    significance=significance if cid == significance_id else None,
                    guard_scores=guard_records if cid == guard_id else (),
                )
                for cid, text, score, flags in scored
            ),
            accepted_version_id=accepted_version_id,
            model_calls_used=self._model.call_count - calls_before,
            elapsed_seconds=time.perf_counter() - t0,
            input_tokens=usage.input_tokens - usage_before.input_tokens,
            output_tokens=usage.output_tokens - usage_before.output_tokens,
        )

    def _build_record(
        self,
        started_at: str,
        baseline_version_id: str,
        baseline_score: float,
        rounds: list[RoundRecord],
        final_score: float,
        stopped_reason: str,
        *,
        final_test_score: float | None = None,
        test_validation_gap: float | None = None,
        validation_overfit: bool = False,
        slice_scores: list[SliceScore] | None = None,
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
            final_test_score=final_test_score,
            test_validation_gap=test_validation_gap,
            validation_overfit=validation_overfit,
            slice_scores=list(slice_scores) if slice_scores else [],
            total_input_tokens=self._counter.usage.input_tokens,
            total_output_tokens=self._counter.usage.output_tokens,
            total_model_calls=self._model.call_count,
            stopped_reason=stopped_reason,
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
        )

    def _score_test(self, final_validation_score: float) -> tuple[float | None, float | None, bool]:
        """Score the final incumbent on the locked test split, exactly once.

        Returns (test_score, validation/test gap, overfit_flag). All-None/False
        when no test split was provided.
        """
        if not self.evalset.test:
            return None, None, False
        self._progress("scoring final incumbent on the locked test split")
        test_score, _, _ = self._score_set(self.artifact.text, self.evalset.test)
        gap = final_validation_score - test_score
        overfit = gap > self.overfit_gap
        if overfit:
            self._progress(
                f"validation/test gap {gap:.4f} exceeds {self.overfit_gap:.4f}: "
                "the validation score may be optimistic (overfit to validation)"
            )
        return test_score, gap, overfit

    def _guard_scores(self, outputs: list[str]) -> list[float]:
        """Aggregate score of each guard over the validation outputs (guard order)."""
        validation = self.evalset.validation
        return [
            guard.aggregate(
                [guard.score(out, ex) for out, ex in zip(outputs, validation, strict=True)]
            )
            for guard in self.guards
        ]

    def _slice_scores(
        self, baseline_val_scores: list[float], final_val_scores: list[float]
    ) -> list[SliceScore]:
        """Per-slice baseline-vs-final scores, grouped by the ``slice_by`` key."""
        slice_by = self.slice_by
        if slice_by is None:
            return []

        def group(scores: list[float]) -> dict[str, list[float]]:
            groups: dict[str, list[float]] = {}
            for example, score in zip(self.evalset.validation, scores, strict=True):
                key = str(example.metadata.get(slice_by, "(unset)"))
                groups.setdefault(key, []).append(score)
            return groups

        base_groups = group(baseline_val_scores)
        final_groups = group(final_val_scores)
        result: list[SliceScore] = []
        for key in sorted(base_groups):
            base = self.objective.aggregate(base_groups[key])
            final = self.objective.aggregate(final_groups[key])
            result.append(SliceScore(slice=key, n=len(base_groups[key]),
                                     baseline_score=base, final_score=final))
            if final < base - self.slice_tolerance:
                self._progress(f"slice {key!r} regressed: {base:.4f} -> {final:.4f}")
        return result

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
