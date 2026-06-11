"""OpenAI backend for the :class:`~recension.models.base.Model` protocol.

Requires the optional extra: ``pip install "recension[openai]"``. The API key is
read from the environment (``OPENAI_API_KEY``) by the OpenAI SDK itself; this
class never accepts a key argument, so a key cannot end up in code or config.

Because it speaks the standard Chat Completions API, this backend also drives
OpenAI-compatible servers (vLLM, LM Studio, OpenRouter, Together, and others)
by passing their ``base_url``.
"""

from __future__ import annotations

from typing import Any

from .base import Message, TokenUsage

__all__ = ["OpenAIModel"]

DEFAULT_MODEL = "gpt-4o-mini"

# DESIGN NOTE: the o-series and gpt-5 reasoning models differ from standard chat
# models on two request fields: they require `max_completion_tokens` (and reject
# `max_tokens`), and they reject a custom `temperature` (only the default is
# allowed). One predicate (`is_reasoning_model`) drives both adaptations;
# `send_temperature` can still override the temperature decision.
_REASONING_PREFIXES = ("o1", "o3", "o4", "gpt-5")


class OpenAIModel:
    """Model backend that calls the OpenAI Chat Completions API.

    Args:
        model: OpenAI model id. Defaults to ``gpt-4o-mini``.
        base_url: Optional base URL for an OpenAI-compatible server. ``None``
            uses the OpenAI API.
        max_retries: Passed through to the SDK client.
        send_temperature: Whether to send ``temperature``. ``None`` (default)
            infers it: reasoning models (o-series, gpt-5) reject a custom value,
            so it is dropped for them and sent otherwise. Pass ``True``/``False``
            to override.
        client: A pre-built OpenAI client to use instead of constructing one
            (for advanced configs such as Azure, or for testing). When given,
            the ``openai`` package is not imported.

    Raises:
        ImportError: If the ``openai`` package is not installed and no ``client``
            is supplied.

    Note:
        ``seed`` is forwarded to the API as a best-effort determinism hint; the
        OpenAI API does not guarantee reproducibility. Deterministic tests use
        :class:`~recension.models.mock.MockModel`, never this backend.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        base_url: str | None = None,
        max_retries: int = 2,
        send_temperature: bool | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.send_temperature = send_temperature
        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "the OpenAI backend needs the 'openai' package; "
                    'install it with: pip install "recension[openai]"'
                ) from exc
            kwargs: dict[str, Any] = {"max_retries": max_retries}
            if base_url is not None:
                kwargs["base_url"] = base_url
            self._client = openai.OpenAI(**kwargs)
        self._calls = 0
        self._last_usage = TokenUsage()

    @property
    def call_count(self) -> int:
        """Number of ``complete`` calls made on this instance."""
        return self._calls

    @property
    def last_usage(self) -> TokenUsage:
        """Token usage of the last call, read from the API response."""
        return self._last_usage

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        """Send ``messages`` to the Chat Completions API and return the reply text."""
        self._calls += 1
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(m) for m in messages],
        }
        # Reasoning models require max_completion_tokens; standard models use max_tokens.
        token_param = "max_completion_tokens" if is_reasoning_model(self.model) else "max_tokens"
        kwargs[token_param] = max_tokens
        if seed is not None:
            kwargs["seed"] = seed
        if should_send_temperature(self.model, self.send_temperature):
            kwargs["temperature"] = temperature
        response = self._client.chat.completions.create(**kwargs)
        self._last_usage = usage_from_response(response)
        return _first_content(response)


def is_reasoning_model(model: str) -> bool:
    """True for OpenAI reasoning models (o-series, gpt-5). Pure; testable offline."""
    return model.startswith(_REASONING_PREFIXES)


def should_send_temperature(model: str, override: bool | None) -> bool:
    """Decide whether to send ``temperature`` for ``model``.

    ``override`` wins when set; otherwise infer from the reasoning-model
    prefixes. Pure and SDK-free so the decision is testable offline.
    """
    if override is not None:
        return override
    return not is_reasoning_model(model)


def _first_content(response: Any) -> str:
    """Text of the first choice, or "" for empty choices / null content (refusals, tool calls)."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    return choices[0].message.content or ""


def usage_from_response(response: Any) -> TokenUsage:
    """Read token usage from a Chat Completions response (SDK-free; testable)."""
    usage = getattr(response, "usage", None)
    return TokenUsage(
        getattr(usage, "prompt_tokens", 0) or 0,
        getattr(usage, "completion_tokens", 0) or 0,
    )
