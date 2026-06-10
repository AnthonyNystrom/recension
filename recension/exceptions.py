"""Exception types for recension.

All library-specific errors derive from :class:`RecensionError` so callers can
catch the whole family with one clause. Measurement-integrity problems get
their own loud types; they are never swallowed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .record import RunRecord

__all__ = [
    "ArtifactError",
    "BudgetExceeded",
    "ConfigError",
    "DegenerateEvalError",
    "LeakageDetected",
    "RecensionError",
]


class RecensionError(Exception):
    """Base class for all recension errors."""


class ArtifactError(RecensionError):
    """Raised for invalid artifact operations (unknown version, no-op commit)."""


class BudgetExceeded(RecensionError):
    """Raised when an optimization run hits ``Budget.max_model_calls``.

    When raised by ``ReflectiveOptimizer.run()``, :attr:`record` carries the
    partial run record up to the point of the overrun, so the audit trail is
    not lost with the failure.
    """

    def __init__(self, message: str, *, record: RunRecord | None = None) -> None:
        super().__init__(message)
        self.record = record


class DegenerateEvalError(RecensionError):
    """Raised when an eval set cannot produce an honest signal.

    Examples: an empty train or validation split, or an example id that
    appears in both splits (which would contaminate the acceptance signal).
    """


class LeakageDetected(RecensionError):
    """Raised in strict mode when a winning candidate trips a leakage heuristic.

    Outside strict mode the same condition is recorded as a flag on the
    candidate instead of raising. See :mod:`recension.leakage`. When raised by
    ``ReflectiveOptimizer.run()``, :attr:`record` carries the partial run
    record so the audit trail survives the failure.
    """

    def __init__(self, message: str, *, record: RunRecord | None = None) -> None:
        super().__init__(message)
        self.record = record


class ConfigError(RecensionError):
    """Raised for invalid or incomplete CLI/run configuration."""
