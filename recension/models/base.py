"""The minimal model interface every backend implements.

The optimizer core is provider-agnostic: it talks to anything satisfying the
:class:`Model` protocol. Backends ship for Anthropic (optional extra) and a
deterministic mock for offline tests.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict, runtime_checkable

__all__ = ["Message", "Model", "Role"]

Role = Literal["system", "user", "assistant"]
"""Message roles understood by every backend."""


class Message(TypedDict):
    """One chat message: a role and its text content."""

    role: Role
    content: str


@runtime_checkable
class Model(Protocol):
    """Narrow protocol for a chat-completion model.

    Implementations must count every completion in :attr:`call_count`; the
    optimizer uses it to enforce ``Budget.max_model_calls``.
    """

    @property
    def call_count(self) -> int:
        """Number of ``complete`` calls made so far on this instance."""
        ...

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> str:
        """Return the model's text completion for ``messages``.

        Args:
            messages: Conversation so far; at most one ``system`` message.
            max_tokens: Upper bound on generated tokens.
            temperature: Sampling temperature; 0 for greedy.
            seed: Optional determinism hint. Backends that cannot honor it
                (e.g. hosted APIs) document that they ignore it.
        """
        ...
