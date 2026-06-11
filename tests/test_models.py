"""Tests for the model layer: protocol conformance, MockModel determinism."""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import pytest

from recension.models import Message, MockModel, Model, Role, SupportsUsage
from recension.models.anthropic import should_send_temperature, split_system


def msg(role: Role, content: str) -> Message:
    return {"role": role, "content": content}


def _installed(name: str) -> bool:
    """True if an (optionally dotted) module is importable; safe when absent."""
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False  # parent package missing (e.g. google.genai with no google)


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
    _installed("anthropic"),
    reason="anthropic extra installed; ImportError path not reachable",
)
def test_anthropic_backend_without_extra_raises_helpful_error() -> None:
    from recension.models.anthropic import AnthropicModel

    with pytest.raises(ImportError, match="recension\\[anthropic\\]"):
        AnthropicModel()


class TestOpenAIModel:
    def test_should_send_temperature(self) -> None:
        from recension.models.openai import should_send_temperature

        assert should_send_temperature("gpt-4o-mini", None) is True
        assert should_send_temperature("o1", None) is False
        assert should_send_temperature("o3-mini", None) is False
        assert should_send_temperature("gpt-5", None) is False
        assert should_send_temperature("o1", True) is True  # override wins
        assert should_send_temperature("gpt-4o", False) is False

    def test_usage_parsing(self) -> None:
        from recension.models.openai import usage_from_response

        resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=12, completion_tokens=5))
        usage = usage_from_response(resp)
        assert (usage.input_tokens, usage.output_tokens) == (12, 5)

    def test_complete_with_injected_client(self) -> None:
        from recension.models.openai import OpenAIModel

        captured: dict[str, object] = {}

        def create(**kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))],
                usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2),
            )

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        model = OpenAIModel(model="gpt-4o-mini", client=fake)
        assert isinstance(model, Model)
        assert isinstance(model, SupportsUsage)
        inputs = [msg("system", "s"), msg("user", "u")]
        snapshot = [dict(m) for m in inputs]
        out = model.complete(inputs, max_tokens=64, seed=9)
        assert out == "hello"
        assert model.call_count == 1
        assert model.last_usage.input_tokens == 7
        assert captured["model"] == "gpt-4o-mini"
        assert captured["messages"] == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
        ]
        assert captured["max_tokens"] == 64  # standard model: max_tokens
        assert "max_completion_tokens" not in captured
        assert captured["seed"] == 9
        assert captured["temperature"] == 0.0  # gpt-4o-mini accepts temperature
        assert inputs == snapshot  # the caller's message dicts were copied, not mutated

    def test_reasoning_model_uses_completion_tokens_and_no_temperature(self) -> None:
        from recension.models.openai import OpenAIModel

        captured: dict[str, object] = {}

        def create(**kwargs: object) -> SimpleNamespace:
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="x"))],
                usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
            )

        fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        OpenAIModel(model="o1", client=fake).complete([msg("user", "hi")], max_tokens=50)
        assert "temperature" not in captured  # reasoning models reject a custom temperature
        assert captured["max_completion_tokens"] == 50  # ...and require this field
        assert "max_tokens" not in captured

    def test_null_content_and_empty_choices_yield_empty_string(self) -> None:
        from recension.models.openai import OpenAIModel

        def fake_with(choices: list[object]) -> SimpleNamespace:
            return SimpleNamespace(
                chat=SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda **k: SimpleNamespace(
                            choices=choices,
                            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=0),
                        )
                    )
                )
            )

        null: list[object] = [SimpleNamespace(message=SimpleNamespace(content=None))]  # refusal
        assert OpenAIModel(client=fake_with(null)).complete([msg("user", "x")]) == ""
        assert OpenAIModel(client=fake_with([])).complete([msg("user", "x")]) == ""

    @pytest.mark.skipif(
        _installed("openai"),
        reason="openai extra installed; ImportError path not reachable",
    )
    def test_without_extra_raises_helpful_error(self) -> None:
        from recension.models.openai import OpenAIModel

        with pytest.raises(ImportError, match="recension\\[openai\\]"):
            OpenAIModel()


class TestGeminiModel:
    def test_to_gemini_contents_maps_roles(self) -> None:
        from recension.models.gemini import to_gemini_contents

        system, contents = to_gemini_contents(
            [msg("system", "a"), msg("system", "b"), msg("user", "hi"), msg("assistant", "ok")]
        )
        assert system == "a\n\nb"
        assert contents == [
            {"role": "user", "parts": [{"text": "hi"}]},
            {"role": "model", "parts": [{"text": "ok"}]},  # assistant -> model
        ]

    def test_usage_parsing(self) -> None:
        from recension.models.gemini import usage_from_response

        resp = SimpleNamespace(
            usage_metadata=SimpleNamespace(prompt_token_count=10, candidates_token_count=4)
        )
        usage = usage_from_response(resp)
        assert (usage.input_tokens, usage.output_tokens) == (10, 4)

    def test_complete_with_injected_client(self) -> None:
        from recension.models.gemini import GeminiModel

        captured: dict[str, object] = {}

        def generate_content(*, model: str, contents: object, config: dict[str, object]) -> object:
            captured["model"] = model
            captured["contents"] = contents
            captured["config"] = config
            return SimpleNamespace(
                text="grounded",
                usage_metadata=SimpleNamespace(prompt_token_count=3, candidates_token_count=1),
            )

        fake = SimpleNamespace(models=SimpleNamespace(generate_content=generate_content))
        model = GeminiModel(model="gemini-2.0-flash", client=fake)
        assert isinstance(model, Model)
        assert isinstance(model, SupportsUsage)
        out = model.complete([msg("system", "sys"), msg("user", "q")], max_tokens=32, seed=5)
        assert out == "grounded"
        assert model.last_usage.input_tokens == 3
        config = captured["config"]
        assert isinstance(config, dict)
        assert config["system_instruction"] == "sys"
        assert config["max_output_tokens"] == 32
        assert config["seed"] == 5

    def test_blocked_response_returns_empty(self) -> None:
        # google-genai's `.text` raises (not returns None) when a candidate has
        # no text part, e.g. a safety-blocked response; we treat it as empty.
        from recension.models.gemini import GeminiModel

        class Blocked:
            usage_metadata = SimpleNamespace(prompt_token_count=2, candidates_token_count=0)

            @property
            def text(self) -> str:
                raise ValueError("no text: response was blocked")

        fake = SimpleNamespace(
            models=SimpleNamespace(generate_content=lambda **k: Blocked())
        )
        assert GeminiModel(client=fake).complete([msg("user", "hi")]) == ""

    @pytest.mark.skipif(
        _installed("google.genai"),
        reason="gemini extra installed; ImportError path not reachable",
    )
    def test_without_extra_raises_helpful_error(self) -> None:
        from recension.models.gemini import GeminiModel

        with pytest.raises(ImportError, match="recension\\[gemini\\]"):
            GeminiModel()


class TestOllamaModel:
    def test_build_payload(self) -> None:
        from recension.models.ollama import build_payload

        payload = build_payload("llama3.2", [msg("user", "hi")], 64, 0.0, 7)
        assert payload["model"] == "llama3.2"
        assert payload["stream"] is False
        assert payload["messages"] == [{"role": "user", "content": "hi"}]
        assert payload["options"] == {"num_predict": 64, "temperature": 0.0, "seed": 7}

    def test_build_payload_omits_seed_when_none(self) -> None:
        from recension.models.ollama import build_payload

        payload = build_payload("m", [msg("user", "x")], 8, 0.5, None)
        assert "seed" not in payload["options"]

    def test_usage_parsing(self) -> None:
        from recension.models.ollama import usage_from_response

        usage = usage_from_response({"prompt_eval_count": 11, "eval_count": 6})
        assert (usage.input_tokens, usage.output_tokens) == (11, 6)

    def test_complete_with_injected_transport(self) -> None:
        from recension.models.ollama import OllamaModel

        seen: dict[str, object] = {}

        def transport(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
            seen["url"] = url
            seen["payload"] = payload
            return {"message": {"content": "local reply"}, "prompt_eval_count": 4, "eval_count": 2}

        model = OllamaModel(model="llama3.2", host="http://localhost:11434/", transport=transport)
        assert isinstance(model, Model)
        assert isinstance(model, SupportsUsage)
        out = model.complete([msg("user", "hi")], max_tokens=16, seed=1)
        assert out == "local reply"
        assert model.call_count == 1
        assert model.last_usage.input_tokens == 4
        assert seen["url"] == "http://localhost:11434/api/chat"  # trailing slash trimmed
