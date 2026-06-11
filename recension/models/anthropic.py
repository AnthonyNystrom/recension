"""Anthropic backend for the :class:`~recension.models.base.Model` protocol.

Requires the optional extra: ``pip install "recension[anthropic]"``. The API
key is read from the environment (``ANTHROPIC_API_KEY``) by the Anthropic SDK
itself; this class never accepts a key argument, so a key cannot end up in
code or config.
"""

from __future__ import annotations

from typing import Any

from .base import Message, TokenUsage

__all__ = ["AnthropicModel"]

DEFAULT_MODEL = "claude-opus-4-8"

# DESIGN NOTE: Opus 4.7+ and Fable 5 reject sampling parameters (temperature
# returns HTTP 400), while older models still accept them. The backend drops
# `temperature` for these families instead of failing the whole run.
_NO_SAMPLING_PREFIXES = ("claude-opus-4-7", "claude-opus-4-8", "claude-fable-5")


class AnthropicModel:
    """Model backend that calls the Anthropic Messages API.

    Args:
        model: Anthropic model id. Defaults to ``claude-opus-4-8``.
        max_retries: Passed through to the SDK client.
        send_temperature: Whether to send the ``temperature`` parameter.
            ``None`` (default) infers it: recent model families (Opus 4.7+,
            Fable 5) reject sampling parameters with HTTP 400, so it is dropped
            for them and sent otherwise. Pass ``True``/``False`` to override the
            inference, an escape hatch for a future model id the built-in
            heuristic does not yet know about.

    Raises:
        ImportError: If the ``anthropic`` package is not installed.

    Note:
        The ``seed`` parameter of :meth:`complete` is ignored, since the Anthropic
        API does not support sampling seeds. Determinism in tests comes from
        :class:`~recension.models.mock.MockModel`, never from this backend.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_retries: int = 2,
        send_temperature: bool | None = None,
    ) -> None:
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "the Anthropic backend needs the 'anthropic' package; "
                'install it with: pip install "recension[anthropic]"'
            ) from exc
        self.model = model
        self.send_temperature = send_temperature
        self._client = anthropic.Anthropic(max_retries=max_retries)
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
        """Send ``messages`` to the Anthropic API and return the reply text.

        ``system`` messages are lifted into the API's ``system`` parameter;
        ``user``/``assistant`` messages pass through in order.
        """
        self._calls += 1
        system, chat = split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": chat,
        }
        if system:
            kwargs["system"] = system
        if should_send_temperature(self.model, self.send_temperature):
            kwargs["temperature"] = temperature
        response = self._client.messages.create(**kwargs)
        usage = getattr(response, "usage", None)
        self._last_usage = TokenUsage(
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
        )
        return "".join(
            block.text for block in response.content if getattr(block, "type", "") == "text"
        )


def should_send_temperature(model: str, override: bool | None) -> bool:
    """Decide whether to send ``temperature`` for ``model``.

    ``override`` wins when set; otherwise infer from the known no-sampling model
    prefixes. Pure and SDK-free so the decision is testable offline.
    """
    if override is not None:
        return override
    return not model.startswith(_NO_SAMPLING_PREFIXES)


def split_system(messages: list[Message]) -> tuple[str, list[dict[str, str]]]:
    """Split a message list into (system text, chat messages).

    Multiple ``system`` messages are joined with blank lines. Exposed as a
    module function so the conversion is testable without the SDK installed.
    """
    system_parts: list[str] = []
    chat: list[dict[str, str]] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
        else:
            chat.append({"role": message["role"], "content": message["content"]})
    return "\n\n".join(system_parts), chat
