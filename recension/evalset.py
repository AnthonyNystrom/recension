"""Held-out evaluation data: examples with an explicit train/validation split.

The split is the integrity backbone of the whole library: the optimizer
diagnoses failures on ``train`` and accepts candidates only on ``validation``.
:class:`EvalSet` enforces the separation at construction time and fails loud
on anything that would corrupt the acceptance signal.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .exceptions import DegenerateEvalError

__all__ = ["EvalSet", "Example"]

_KNOWN_KEYS = {"id", "input", "expected", "rubric", "split"}


@dataclass(frozen=True)
class Example:
    """One evaluation example.

    Attributes:
        id: Stable identifier; referenced by diagnoses and provenance.
        input: The text given to the model (alongside the artifact).
        expected: Gold output for reference-based objectives, if any.
        rubric: Per-example grading rubric for model-graded objectives, if any.
        metadata: Any extra fields the objective may need.
    """

    id: str
    input: str
    expected: str | None = None
    rubric: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class EvalSet:
    """Examples partitioned into ``train`` and ``validation``.

    Raises:
        DegenerateEvalError: If either split is empty, an id is duplicated
            within a split, or an id appears in both splits.
    """

    def __init__(self, train: Sequence[Example], validation: Sequence[Example]) -> None:
        if not train:
            raise DegenerateEvalError("train split is empty")
        if not validation:
            raise DegenerateEvalError("validation split is empty")
        train_ids = [e.id for e in train]
        val_ids = [e.id for e in validation]
        for label, ids in (("train", train_ids), ("validation", val_ids)):
            if len(set(ids)) != len(ids):
                raise DegenerateEvalError(f"duplicate example ids in {label} split")
        overlap = set(train_ids) & set(val_ids)
        if overlap:
            raise DegenerateEvalError(
                f"example ids appear in both splits (would contaminate acceptance): "
                f"{sorted(overlap)}"
            )
        self._train = tuple(train)
        self._validation = tuple(validation)

    @property
    def train(self) -> tuple[Example, ...]:
        """Examples used to diagnose failures."""
        return self._train

    @property
    def validation(self) -> tuple[Example, ...]:
        """Held-out examples used to accept or reject candidates."""
        return self._validation

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> EvalSet:
        """Build an eval set from dict records.

        Each record needs ``id``, ``input``, and ``split`` (``"train"`` or
        ``"validation"``), plus optional ``expected`` and ``rubric``. Unknown
        keys land in :attr:`Example.metadata`.

        Raises:
            DegenerateEvalError: On a missing key, an unknown split value, or
                any split-integrity violation.
        """
        train: list[Example] = []
        validation: list[Example] = []
        for i, record in enumerate(records):
            try:
                example_id = str(record["id"])
                example_input = str(record["input"])
                split = record["split"]
            except KeyError as exc:
                raise DegenerateEvalError(f"record {i} is missing required key {exc}") from exc
            example = Example(
                id=example_id,
                input=example_input,
                expected=None if record.get("expected") is None else str(record["expected"]),
                rubric=None if record.get("rubric") is None else str(record["rubric"]),
                metadata={k: v for k, v in record.items() if k not in _KNOWN_KEYS},
            )
            if split == "train":
                train.append(example)
            elif split == "validation":
                validation.append(example)
            else:
                raise DegenerateEvalError(
                    f"record {example_id!r} has unknown split {split!r} "
                    "(expected 'train' or 'validation')"
                )
        return cls(train, validation)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalSet:
        """Build an eval set from a JSONL file of records (see ``from_records``).

        Raises:
            DegenerateEvalError: On a line that is not valid JSON (with the
                file and line number), or any split-integrity violation.
        """
        records = []
        p = Path(path)
        with p.open(encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise DegenerateEvalError(f"{p} line {lineno}: invalid JSON: {exc}") from exc
        return cls.from_records(records)

    def __repr__(self) -> str:
        return f"EvalSet(train={len(self._train)}, validation={len(self._validation)})"
