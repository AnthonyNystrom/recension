"""Tests for failure diagnosis and distinct-candidate generation."""

from __future__ import annotations

from recension import Example
from recension.models import Message, MockModel
from recension.proposer import FailureCase, diagnose, extract_candidate, propose


def failure(example_id: str = "t1") -> FailureCase:
    return FailureCase(
        example=Example(id=example_id, input="some input", expected="gold"),
        output="bad output",
        score=0.0,
    )


class TestExtractCandidate:
    def test_extracts_tagged_text(self) -> None:
        reply = "<revised_artifact>\nNew prompt text.\n</revised_artifact>"
        assert extract_candidate(reply) == "New prompt text."

    def test_falls_back_to_whole_reply(self) -> None:
        assert extract_candidate("  Just the text.  ") == "Just the text."

    def test_multiline_candidate(self) -> None:
        reply = "<revised_artifact>\nline one\nline two\n</revised_artifact>"
        assert extract_candidate(reply) == "line one\nline two"


class TestDiagnose:
    def test_includes_artifact_and_failures_in_prompt(self) -> None:
        seen: list[list[Message]] = []

        def script(messages: list[Message]) -> str:
            seen.append(messages)
            return "  the artifact is ambiguous  "

        result = diagnose(MockModel(script=script), "ARTIFACT TEXT", [failure("ex-9")])
        assert result == "the artifact is ambiguous"
        user_content = seen[0][1]["content"]
        assert "ARTIFACT TEXT" in user_content
        assert "ex-9" in user_content
        assert "bad output" in user_content
        assert "gold" in user_content


class TestPropose:
    def test_generates_n_distinct_candidates(self) -> None:
        variants = iter(
            [
                "State the allowed labels explicitly in the prompt.",
                "Show two worked examples before asking for an answer.",
                "Demand a one-word reply and nothing else.",
            ]
        )

        def script(messages: list[Message]) -> str:
            return f"<revised_artifact>\n{next(variants)}\n</revised_artifact>"

        out = propose(MockModel(script=script), "incumbent", "diag", 3)
        assert len(out) == 3
        assert len(set(out)) == 3

    def test_near_duplicates_are_rejected_and_regenerated(self) -> None:
        replies = iter(
            [
                "<revised_artifact>\nAlways answer with one single word.\n</revised_artifact>",
                # near-duplicate of the first (one char changed)
                "<revised_artifact>\nAlways answer with one single word!\n</revised_artifact>",
                "<revised_artifact>\nState the allowed labels explicitly.\n</revised_artifact>",
            ]
        )
        out = propose(MockModel(script=lambda m: next(replies)), "incumbent", "diag", 2)
        assert out == [
            "Always answer with one single word.",
            "State the allowed labels explicitly.",
        ]

    def test_candidate_identical_to_incumbent_is_rejected(self) -> None:
        replies = iter(
            [
                "<revised_artifact>\nincumbent text\n</revised_artifact>",
                "<revised_artifact>\na genuinely different revision\n</revised_artifact>",
            ]
        )
        out = propose(MockModel(script=lambda m: next(replies)), "incumbent text", "diag", 1)
        assert out == ["a genuinely different revision"]

    def test_returns_shortfall_when_model_keeps_duplicating(self) -> None:
        model = MockModel(script=lambda m: "<revised_artifact>\nsame thing\n</revised_artifact>")
        out = propose(model, "incumbent", "diag", 3)
        assert out == ["same thing"]
        # attempts are capped, not infinite
        assert model.call_count == 9

    def test_prompt_asks_for_distinct_approach_after_first_candidate(self) -> None:
        seen: list[str] = []
        variants = iter(
            [
                "State the allowed labels explicitly in the prompt.",
                "Show two worked examples before asking for an answer.",
            ]
        )

        def script(messages: list[Message]) -> str:
            seen.append(messages[1]["content"])
            return f"<revised_artifact>\n{next(variants)}\n</revised_artifact>"

        propose(MockModel(script=script), "incumbent", "diag", 2)
        assert "genuinely different approach" not in seen[0]
        assert "genuinely different approach" in seen[1]
        assert "State the allowed labels explicitly" in seen[1]
