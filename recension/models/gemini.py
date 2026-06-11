"""Google Gemini backend for the :class:`~recension.models.base.Model` protocol.

Requires the optional extra: ``pip install "recension[gemini]"``. The API key is
read from the environment (``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``) by the
google-genai SDK itself; this class never accepts a key argument, so a key
cannot end up in code or config.
"""

from __future__ import annotations

from typing import Any

from .base import Message, TokenUsage

__all__ = ["GeminiModel"]

DEFAULT_MODEL = "gemini-2.0-flash"


class GeminiModel:
    """Model backend that calls the Google Gemini API (google-genai SDK).

    Args:
        model: Gemini model id. Defaults to ``gemini-2.0-flash``.
        client: A pre-built ``google.genai.Client`` to use instead of
            constructing one (for advanced configs or testing). When given, the
            ``google-genai`` package is not imported.

    Raises:
        ImportError: If the ``google-genai`` package is not installed and no
            ``client`` is supplied.

    Note:
        ``seed`` is forwarded as a best-effort determinism hint; the API does not
        guarantee reproducibility. Deterministic tests use
        :class:`~recension.models.mock.MockModel`, never this backend.
    """

    def __init__(self, model: str = DEFAULT_MODEL, *, client: Any = None) -> None:
        self.model = model
        if client is not None:
            self._client = client
        else:
            try:
                from google import genai
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise ImportError(
                    "the Gemini backend needs the 'google-genai' package; "
                    'install it with: pip install "recension[gemini]"'
                ) from exc
            self._client = genai.Client()
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
        """Send ``messages`` to the Gemini API and return the reply text.

        ``system`` messages become the ``system_instruction``; ``assistant``
        messages are mapped to Gemini's ``model`` role.
        """
        self._calls += 1
        system, contents = to_gemini_contents(messages)
        config: dict[str, Any] = {"max_output_tokens": max_tokens, "temperature": temperature}
        if system:
            config["system_instruction"] = system
        if seed is not None:
            config["seed"] = seed
        response = self._client.models.generate_content(
            model=self.model, contents=contents, config=config
        )
        self._last_usage = usage_from_response(response)
        # `.text` is a property that raises (not returns None) when the candidate
        # has no text part, e.g. a safety-blocked response; treat that as empty.
        try:
            text = response.text
        except Exception:
            text = None
        return text or ""


def to_gemini_contents(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Split messages into (system instruction, Gemini ``contents``).

    ``system`` messages are joined into the instruction; ``user`` stays ``user``
    and ``assistant`` becomes ``model``. Pure and SDK-free so the mapping is
    testable offline.
    """
    system_parts: list[str] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        if message["role"] == "system":
            system_parts.append(message["content"])
            continue
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": message["content"]}]})
    return "\n\n".join(system_parts), contents


def usage_from_response(response: Any) -> TokenUsage:
    """Read token usage from a Gemini response (SDK-free; testable)."""
    usage = getattr(response, "usage_metadata", None)
    return TokenUsage(
        getattr(usage, "prompt_token_count", 0) or 0,
        getattr(usage, "candidates_token_count", 0) or 0,
    )
