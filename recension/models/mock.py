"""Deterministic mock model for offline tests and reproducible examples.

The entire test suite runs against :class:`MockModel`: no network, no API
key. Given the same messages, seed, and script, it always returns the same
output, which is what makes seeded optimizer runs reproducible.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable

from .base import Message

__all__ = ["MockModel"]


class MockModel:
    """A deterministic, scriptable stand-in for a real model.

    Args:
        script: Optional callable mapping the message list to a reply. Use it
            to simulate task answers, diagnoses, judges, or candidate
            proposals in tests and examples. When omitted, replies are
            deterministic pseudo-text derived from a hash of the messages
            and seed.
        seed: Folded into the unscripted reply hash, so different seeds give
            different (but stable) outputs.
    """

    def __init__(
        self,
        script: Callable[[list[Message]], str] | None = None,
        *,
        seed: int = 0,
    ) -> None:
        self.script = script
        self.seed = seed
        self._calls = 0

    @property
    def call_count(self) -> int:
        """Number of ``complete`` calls made on this instance."""
        return self._calls

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        """Return a deterministic reply for ``messages``.

        The ``seed`` argument, when given, overrides the instance seed for
        this call. ``max_tokens`` and ``temperature`` are accepted for
        protocol compatibility; they do not change the output.
        """
        self._calls += 1
        if self.script is not None:
            return self.script(list(messages))
        effective_seed = self.seed if seed is None else seed
        digest = hashlib.sha256()
        digest.update(str(effective_seed).encode())
        for message in messages:
            digest.update(message["role"].encode())
            digest.update(b"\x00")
            digest.update(message["content"].encode())
            digest.update(b"\x01")
        return f"mock-output-{digest.hexdigest()[:16]}"
