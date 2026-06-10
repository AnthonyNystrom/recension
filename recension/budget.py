"""The update-time compute dial.

Every knob the optimizer spends model calls on is here, caller-controlled.
Nothing about update-time compute is hardcoded in the loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["Budget"]


@dataclass(frozen=True)
class Budget:
    """Caller-controlled limits on update-time compute.

    Attributes:
        candidates_per_round: Distinct candidate edits generated per round.
        rounds: Maximum optimization rounds.
        diagnosis_depth: How many failed train examples are analyzed per round.
        max_model_calls: Hard ceiling on total model calls for the run
            (task scoring, diagnosis, proposals, and judge calls all count).
            ``None`` means unlimited. The optimizer raises
            :class:`~recension.exceptions.BudgetExceeded` when the ceiling
            would be crossed.
    """

    candidates_per_round: int = 4
    rounds: int = 3
    diagnosis_depth: int = 1
    max_model_calls: int | None = None

    def __post_init__(self) -> None:
        if self.candidates_per_round < 1:
            raise ValueError("candidates_per_round must be >= 1")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.diagnosis_depth < 1:
            raise ValueError("diagnosis_depth must be >= 1")
        if self.max_model_calls is not None and self.max_model_calls < 1:
            raise ValueError("max_model_calls must be >= 1 or None")

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form for embedding in a run record."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Budget:
        """Inverse of :meth:`to_dict`."""
        return cls(**data)
