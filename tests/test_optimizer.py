"""Optimizer tests, including the end-to-end anchor test.

Everything runs against scripted MockModels — no network, fully deterministic.
"""

from __future__ import annotations

import re

import pytest

from recension import (
    Budget,
    BudgetExceeded,
    EvalSet,
    ExactMatch,
    LeakageDetected,
    LLMJudge,
    ReflectiveOptimizer,
    RunRecord,
    TextArtifact,
)
from recension.models import Message, MockModel

GOOD_INSTRUCTION = "Reply with exactly one word: positive or negative."

POSITIVE_WORDS = ("love", "great", "wonderful")
NEGATIVE_WORDS = ("terrible", "awful", "hate")


def sentiment_evalset() -> EvalSet:
    return EvalSet.from_records(
        [
            {"id": "t1", "input": "I love this thing", "expected": "positive", "split": "train"},
            {"id": "t2", "input": "this is terrible", "expected": "negative", "split": "train"},
            {"id": "t3", "input": "what a great day", "expected": "positive", "split": "train"},
            {"id": "v1", "input": "I hate waiting", "expected": "negative", "split": "validation"},
            {"id": "v2", "input": "wonderful place", "expected": "positive", "split": "validation"},
            {"id": "v3", "input": "awful noise", "expected": "negative", "split": "validation"},
        ]
    )


def _label_for(text: str) -> str:
    if any(w in text for w in POSITIVE_WORDS):
        return "positive"
    if any(w in text for w in NEGATIVE_WORDS):
        return "negative"
    return "unknown"


def sentiment_script(messages: list[Message]) -> str:
    """Scripted model: tasks succeed only once the artifact states the labels;
    diagnosis and proposals are deterministic; proposal #2 is the real fix."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next(m["content"] for m in messages if m["role"] == "user")
    if system.startswith("You analyze why"):
        return "The artifact never states the allowed labels."
    if system.startswith("You revise text artifacts"):
        index = int(re.search(r"Propose revision (\d+)", user).group(1))  # type: ignore[union-attr]
        artifact = re.search(r"<artifact>\n(.*?)\n</artifact>", user, re.DOTALL).group(1)  # type: ignore[union-attr]
        variants = {
            1: "Answer about the sentiment, briefly.",
            2: artifact + "\n" + GOOD_INSTRUCTION,
            3: "Be concise. Use only one word in your reply.",
            4: "Think carefully, then answer in a word.",
        }
        return f"<revised_artifact>\n{variants[index]}\n</revised_artifact>"
    # task call
    if GOOD_INSTRUCTION in system:
        return _label_for(user)
    return "neutral"


def run_sentiment(seed: int = 7) -> RunRecord:
    optimizer = ReflectiveOptimizer(
        artifact=TextArtifact.from_text("Label the sentiment of the message.", name="clf"),
        evalset=sentiment_evalset(),
        objective=ExactMatch(),
        model=MockModel(script=sentiment_script),
        budget=Budget(candidates_per_round=4, rounds=3, diagnosis_depth=2),
        seed=seed,
    )
    return optimizer.run()


class TestEndToEnd:
    """The anchor test: a full run() against the mock, record verified end to end."""

    def test_accepts_the_real_fix_and_records_everything(self) -> None:
        record = run_sentiment()

        # Baseline: the vague artifact scores zero on validation.
        assert record.baseline_score == 0.0

        # Round 1 accepted the candidate that states the labels.
        round1 = record.rounds[0]
        assert round1.diagnosis == "The artifact never states the allowed labels."
        assert round1.failure_example_ids == ("t1", "t2")
        assert len(round1.candidates) == 4
        accepted = [c for c in round1.candidates if c.accepted]
        assert len(accepted) == 1
        assert GOOD_INSTRUCTION in accepted[0].text
        assert accepted[0].validation_score == 1.0
        assert round1.accepted_version_id is not None
        assert f"+{GOOD_INSTRUCTION}" in accepted[0].diff

        # The three rejected candidates are recorded with their scores.
        rejected = [c for c in round1.candidates if not c.accepted]
        assert len(rejected) == 3
        assert all(c.validation_score == 0.0 for c in rejected)

        # Round 2 found no further improvement and the run stopped honestly.
        assert len(record.rounds) == 2
        assert record.rounds[1].accepted_version_id is None
        assert record.stopped_reason == "no_improvement"
        assert record.final_score == 1.0

        # The embedded artifact carries full provenance for the accepted version.
        artifact = record.restored_artifact()
        assert artifact.text.endswith(GOOD_INSTRUCTION)
        provenance = artifact.current().provenance
        assert provenance is not None
        assert provenance.diagnosis == round1.diagnosis
        assert provenance.incumbent_score == 0.0
        assert provenance.candidate_score == 1.0
        assert len(provenance.rejected_candidates) == 3
        assert provenance.diff

        # Bookkeeping is honest.
        assert record.final_version_id == round1.accepted_version_id
        assert record.total_model_calls == sum(r.model_calls_used for r in record.rounds) + len(
            sentiment_evalset().validation
        )
        assert record.objective_name == "exact_match"
        assert record.model_graded is False
        assert record.stopped_reason in ("completed", "no_improvement")

    def test_accepted_candidate_records_its_train_score(self) -> None:
        record = run_sentiment()
        accepted = [c for r in record.rounds for c in r.candidates if c.accepted]
        assert len(accepted) == 1
        # The winner states the labels, so it scores perfectly on train too.
        assert accepted[0].train_score == 1.0
        rejected = [c for r in record.rounds for c in r.candidates if not c.accepted]
        assert rejected and all(c.train_score is None for c in rejected)
        # And the train score survives a serialization round trip.
        assert RunRecord.from_json(record.to_json()) == record

    def test_accepted_incumbent_train_score_is_reused_not_rescored(self) -> None:
        lines: list[str] = []
        ReflectiveOptimizer(
            artifact=TextArtifact.from_text("Label the sentiment of the message.", name="clf"),
            evalset=sentiment_evalset(),
            objective=ExactMatch(),
            model=MockModel(script=sentiment_script),
            budget=Budget(candidates_per_round=4, rounds=2, diagnosis_depth=2),
            seed=7,
            stop_on_no_improvement=False,
            on_progress=lines.append,
        ).run()
        joined = "\n".join(lines)
        assert "round 1: scoring incumbent on train" in joined
        # Round 2's incumbent is round 1's accepted winner — its train score is
        # cached, so it is reused rather than re-scored.
        assert "round 2: reusing cached incumbent train score" in joined
        assert "round 2: scoring incumbent on train" not in joined

    def test_record_serialization_roundtrip(self) -> None:
        record = run_sentiment()
        assert RunRecord.from_json(record.to_json()) == record

    def test_seeded_runs_are_reproducible(self) -> None:
        a, b = run_sentiment(seed=7), run_sentiment(seed=7)
        assert a.final_version_id == b.final_version_id
        assert a.baseline_score == b.baseline_score
        assert a.final_score == b.final_score
        assert a.total_model_calls == b.total_model_calls
        assert [c.text for r in a.rounds for c in r.candidates] == [
            c.text for r in b.rounds for c in r.candidates
        ]
        assert [r.diagnosis for r in a.rounds] == [r.diagnosis for r in b.rounds]


class TestBudget:
    def test_max_model_calls_raises_with_partial_record(self) -> None:
        optimizer = ReflectiveOptimizer(
            artifact=TextArtifact.from_text("Label the sentiment.", name="clf"),
            evalset=sentiment_evalset(),
            objective=ExactMatch(),
            model=MockModel(script=sentiment_script),
            budget=Budget(max_model_calls=2),  # baseline scoring alone needs 3
        )
        with pytest.raises(BudgetExceeded) as excinfo:
            optimizer.run()
        record = excinfo.value.record
        assert record is not None
        assert record.stopped_reason == "budget_exceeded"
        assert record.total_model_calls <= 2

    def test_ceiling_is_never_crossed(self) -> None:
        model = MockModel(script=sentiment_script)
        optimizer = ReflectiveOptimizer(
            artifact=TextArtifact.from_text("Label the sentiment.", name="clf"),
            evalset=sentiment_evalset(),
            objective=ExactMatch(),
            model=model,
            budget=Budget(max_model_calls=10),
        )
        with pytest.raises(BudgetExceeded):
            optimizer.run()
        assert model.call_count <= 10

    def test_midround_overrun_preserves_the_partial_round(self) -> None:
        # baseline 3 + train 3 + diagnose 1 + propose 4 = 11; candidate r1-c1
        # scoring is 3 more (counts 12-14); r1-c2's first call hits the ceiling.
        optimizer = ReflectiveOptimizer(
            artifact=TextArtifact.from_text("Label the sentiment.", name="clf"),
            evalset=sentiment_evalset(),
            objective=ExactMatch(),
            model=MockModel(script=sentiment_script),
            budget=Budget(candidates_per_round=4, rounds=3, diagnosis_depth=2,
                          max_model_calls=14),
            seed=7,
        )
        with pytest.raises(BudgetExceeded) as excinfo:
            optimizer.run()
        record = excinfo.value.record
        assert record is not None
        assert record.stopped_reason == "budget_exceeded"
        # The in-progress round is preserved, with its diagnosis and the one
        # candidate scored before the budget ran out.
        last = record.rounds[-1]
        assert last.diagnosis == "The artifact never states the allowed labels."
        assert [c.candidate_id for c in last.candidates] == ["r1-c1"]

    def test_model_graded_judge_calls_count_against_budget(self) -> None:
        # Each validation example costs a task call AND a judge call. With only
        # 2 validation examples and a ceiling of 3, baseline scoring alone (task,
        # judge, task, judge...) overruns — which can only happen if judge calls
        # are counted. Task-only counting would not reach 3 during baseline.
        def judge_script(messages: list[Message]) -> str:
            user = next(m["content"] for m in messages if m["role"] == "user")
            if user.startswith("You are grading a model output"):
                return "7"
            return "an answer"

        model = MockModel(script=judge_script)
        evalset = EvalSet.from_records(
            [
                {"id": "t1", "input": "a", "split": "train"},
                {"id": "v1", "input": "b", "split": "validation"},
                {"id": "v2", "input": "c", "split": "validation"},
            ]
        )
        optimizer = ReflectiveOptimizer(
            artifact=TextArtifact.from_text("Summarize the report.", name="s"),
            evalset=evalset,
            objective=LLMJudge(model, rubric="be useful"),
            model=model,
            budget=Budget(max_model_calls=3),
        )
        with pytest.raises(BudgetExceeded) as excinfo:
            optimizer.run()
        assert model.call_count <= 3
        record = excinfo.value.record
        assert record is not None
        assert record.model_graded is True
        assert record.total_model_calls == 3
        # The caller's objective is left untouched after the scoped gate.
        assert optimizer.objective.model is model  # type: ignore[attr-defined]


class TestNoImprovement:
    @staticmethod
    def hopeless_script(messages: list[Message]) -> str:
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user = next(m["content"] for m in messages if m["role"] == "user")
        if system.startswith("You analyze why"):
            return "no idea"
        if system.startswith("You revise text artifacts"):
            index = int(re.search(r"Propose revision (\d+)", user).group(1))  # type: ignore[union-attr]
            return f"<revised_artifact>\nhopeless variant number {index}\n</revised_artifact>"
        return "always wrong"

    def test_stops_after_first_flat_round_by_default(self) -> None:
        record = ReflectiveOptimizer(
            artifact=TextArtifact.from_text("x", name="a"),
            evalset=sentiment_evalset(),
            objective=ExactMatch(),
            model=MockModel(script=self.hopeless_script),
            budget=Budget(candidates_per_round=2, rounds=3),
        ).run()
        assert len(record.rounds) == 1
        assert record.stopped_reason == "no_improvement"
        assert record.final_version_id == record.baseline_version_id
        assert record.final_score == record.baseline_score

    def test_continue_policy_spends_all_rounds(self) -> None:
        record = ReflectiveOptimizer(
            artifact=TextArtifact.from_text("x", name="a"),
            evalset=sentiment_evalset(),
            objective=ExactMatch(),
            model=MockModel(script=self.hopeless_script),
            budget=Budget(candidates_per_round=2, rounds=3),
            stop_on_no_improvement=False,
        ).run()
        assert len(record.rounds) == 3
        assert record.stopped_reason == "completed"


def overfit_evalset() -> EvalSet:
    return EvalSet.from_records(
        [
            {"id": "t1", "input": "plain alpha", "expected": "yes", "split": "train"},
            {"id": "t2", "input": "plain beta", "expected": "yes", "split": "train"},
            {"id": "v1", "input": "shiny gamma", "expected": "yes", "split": "validation"},
            {"id": "v2", "input": "shiny delta", "expected": "yes", "split": "validation"},
        ]
    )


def overfitting_script(messages: list[Message]) -> str:
    """Proposal #1 'fixes' only validation examples — a validation-only jump."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next(m["content"] for m in messages if m["role"] == "user")
    if system.startswith("You analyze why"):
        return "unclear"
    if system.startswith("You revise text artifacts"):
        index = int(re.search(r"Propose revision (\d+)", user).group(1))  # type: ignore[union-attr]
        variants = {
            1: "Use the VALIDATION_TRICK to answer.",
            2: "A harmless alternative wording.",
        }
        return f"<revised_artifact>\n{variants[index]}\n</revised_artifact>"
    if "VALIDATION_TRICK" in system and "shiny" in user:
        return "yes"
    return "wrong"


class TestLeakage:
    def test_strict_mode_raises_with_record(self) -> None:
        optimizer = ReflectiveOptimizer(
            artifact=TextArtifact.from_text("Answer yes or no.", name="a"),
            evalset=overfit_evalset(),
            objective=ExactMatch(),
            model=MockModel(script=overfitting_script),
            budget=Budget(candidates_per_round=2, rounds=1),
            strict_leakage=True,
        )
        with pytest.raises(LeakageDetected, match="implausible_gain") as excinfo:
            optimizer.run()
        record = excinfo.value.record
        assert record is not None
        assert record.stopped_reason == "leakage_detected"
        assert len(record.rounds) == 1
        # nothing was committed
        assert record.rounds[0].accepted_version_id is None
        assert record.final_version_id == record.baseline_version_id
        flagged = [c for c in record.rounds[0].candidates if c.leakage_flags]
        assert flagged and "implausible_gain" in flagged[0].leakage_flags[0]

    def test_default_mode_accepts_but_surfaces_flags(self) -> None:
        record = ReflectiveOptimizer(
            artifact=TextArtifact.from_text("Answer yes or no.", name="a"),
            evalset=overfit_evalset(),
            objective=ExactMatch(),
            model=MockModel(script=overfitting_script),
            budget=Budget(candidates_per_round=2, rounds=1),
        ).run()
        round1 = record.rounds[0]
        assert round1.accepted_version_id is not None
        accepted = next(c for c in round1.candidates if c.accepted)
        assert any("implausible_gain" in flag for flag in accepted.leakage_flags)


class TestProgress:
    def test_progress_callback_receives_key_events(self) -> None:
        lines: list[str] = []
        ReflectiveOptimizer(
            artifact=TextArtifact.from_text("Label the sentiment.", name="clf"),
            evalset=sentiment_evalset(),
            objective=ExactMatch(),
            model=MockModel(script=sentiment_script),
            budget=Budget(candidates_per_round=4, rounds=1),
            on_progress=lines.append,
        ).run()
        joined = "\n".join(lines)
        assert "baseline validation score" in joined
        assert "diagnosis" in joined
        assert "accepted" in joined
