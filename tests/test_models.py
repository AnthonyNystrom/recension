"""Tests for the model layer: protocol conformance, MockModel determinism."""

from __future__ import annotations

import importlib.util

import pytest

from recension.models import Message, MockModel, Model, Role
from recension.models.anthropic import should_send_temperature, split_system


def msg(role: Role, content: str) -> Message:
    return {"role": role, "content": content}


class TestMockModel:
    def test_satisfies_protocol(self) -> None:
        assert isinstance(MockModel(), Model)

    def test_deterministic_given_same_inputs(self) -> None:
        a = MockModel(seed=7).complete([msg("user", "hi")])
        b = MockModel(seed=7).complete([msg("user", "hi")])
        assert a == b

    def test_seed_changes_output(self) -> None:
        a = MockModel(seed=1).complete([msg("user", "hi")])
        b = MockModel(seed=2).complete([msg("user", "hi")])
        assert a != b

    def test_call_seed_overrides_instance_seed(self) -> None:
        model = MockModel(seed=1)
        assert model.complete([msg("user", "hi")], seed=2) == MockModel(seed=2).complete(
            [msg("user", "hi")]
        )

    def test_messages_change_output(self) -> None:
        model = MockModel()
        assert model.complete([msg("user", "a")]) != model.complete([msg("user", "b")])

    def test_counts_calls(self) -> None:
        model = MockModel()
        assert model.call_count == 0
        model.complete([msg("user", "x")])
        model.complete([msg("user", "y")])
        assert model.call_count == 2

    def test_script_drives_output(self) -> None:
        model = MockModel(script=lambda messages: f"saw {len(messages)} messages")
        assert model.complete([msg("system", "s"), msg("user", "u")]) == "saw 2 messages"
        assert model.call_count == 1

    def test_reports_synthetic_usage(self) -> None:
        from recension.models import SupportsUsage

        model = MockModel(script=lambda m: "abcdefgh")  # 8-char reply
        assert isinstance(model, SupportsUsage)
        model.complete([msg("user", "x" * 40)])
        assert model.last_usage.input_tokens == 10  # 40 chars // 4
        assert model.last_usage.output_tokens == 2  # 8 chars // 4


class TestSplitSystem:
    def test_lifts_system_messages(self) -> None:
        system, chat = split_system(
            [msg("system", "rules"), msg("user", "hi"), msg("assistant", "hello")]
        )
        assert system == "rules"
        assert chat == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]

    def test_joins_multiple_system_messages(self) -> None:
        system, chat = split_system([msg("system", "a"), msg("system", "b"), msg("user", "u")])
        assert system == "a\n\nb"
        assert len(chat) == 1

    def test_no_system(self) -> None:
        system, chat = split_system([msg("user", "u")])
        assert system == ""
        assert chat == [{"role": "user", "content": "u"}]


class TestShouldSendTemperature:
    def test_inferred_off_for_no_sampling_families(self) -> None:
        assert should_send_temperature("claude-opus-4-8", None) is False
        assert should_send_temperature("claude-fable-5", None) is False

    def test_inferred_on_for_older_families(self) -> None:
        assert should_send_temperature("claude-sonnet-4-5", None) is True
        assert should_send_temperature("claude-opus-4-1", None) is True

    def test_override_wins_either_way(self) -> None:
        # Force-send for an unknown future id the heuristic would drop...
        assert should_send_temperature("claude-opus-4-8", True) is True
        # ...or force-drop for one it would otherwise send.
        assert should_send_temperature("claude-sonnet-4-5", False) is False


@pytest.mark.skipif(
    importlib.util.find_spec("anthropic") is not None,
    reason="anthropic extra installed; ImportError path not reachable",
)
def test_anthropic_backend_without_extra_raises_helpful_error() -> None:
    from recension.models.anthropic import AnthropicModel

    with pytest.raises(ImportError, match="recension\\[anthropic\\]"):
        AnthropicModel()
