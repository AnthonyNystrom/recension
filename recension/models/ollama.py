"""Ollama backend for the :class:`~recension.models.base.Model` protocol.

Runs local models served by `Ollama <https://ollama.com>`_. Needs no extra
dependency: it talks to the local HTTP API with the standard library. Point it
at a running Ollama server (default ``http://localhost:11434``) and pull the
model first (``ollama pull llama3.2``).
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable
from typing import Any

from .base import Message, TokenUsage

__all__ = ["OllamaModel"]

DEFAULT_MODEL = "llama3.2"

Transport = Callable[[str, "dict[str, Any]", float], "dict[str, Any]"]


class OllamaModel:
    """Model backend that calls a local Ollama server's chat API.

    Args:
        model: Ollama model name (must be pulled locally). Defaults to
            ``llama3.2``.
        host: Base URL of the Ollama server.
        timeout: Per-request timeout in seconds.
        transport: Override the HTTP transport ``(url, payload, timeout) ->
            response_dict``; used for testing. Defaults to a stdlib ``urllib``
            POST.

    Note:
        ``seed`` is forwarded to Ollama; local models honor it fairly well, but
        deterministic tests still use :class:`~recension.models.mock.MockModel`,
        never a live server.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        host: str = "http://localhost:11434",
        timeout: float = 120.0,
        transport: Transport | None = None,
    ) -> None:
        self.model = model
        self._url = host.rstrip("/") + "/api/chat"
        self._timeout = timeout
        self._transport: Transport = transport or _http_post
        self._calls = 0
        self._last_usage = TokenUsage()

    @property
    def call_count(self) -> int:
        """Number of ``complete`` calls made on this instance."""
        return self._calls

    @property
    def last_usage(self) -> TokenUsage:
        """Token usage of the last call, read from the Ollama response."""
        return self._last_usage

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        """Send ``messages`` to the Ollama chat API and return the reply text."""
        self._calls += 1
        payload = build_payload(self.model, messages, max_tokens, temperature, seed)
        data = self._transport(self._url, payload, self._timeout)
        self._last_usage = usage_from_response(data)
        return data.get("message", {}).get("content", "") or ""


def build_payload(
    model: str,
    messages: list[Message],
    max_tokens: int,
    temperature: float,
    seed: int | None,
) -> dict[str, Any]:
    """Build the Ollama ``/api/chat`` request body (pure; testable)."""
    options: dict[str, Any] = {"num_predict": max_tokens, "temperature": temperature}
    if seed is not None:
        options["seed"] = seed
    return {
        "model": model,
        "messages": [dict(m) for m in messages],
        "stream": False,
        "options": options,
    }


def usage_from_response(data: dict[str, Any]) -> TokenUsage:
    """Read token usage from an Ollama response (pure; testable)."""
    return TokenUsage(
        int(data.get("prompt_eval_count", 0) or 0),
        int(data.get("eval_count", 0) or 0),
    )


def _http_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (local API)
        result: dict[str, Any] = json.loads(response.read().decode("utf-8"))
    return result
