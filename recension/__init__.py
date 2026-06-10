"""recension: measured optimization of the text layer around a language model.

Versioned artifacts with provenance, held-out evaluation, leakage detection,
and a complete audit record. See the README for the prior-art boundary:
optimization mechanics are well covered by DSPy and GEPA; recension's
contribution is the measurement-and-governance shell around a text artifact.
"""

from __future__ import annotations

from .artifact import Provenance, RejectedCandidate, TextArtifact, Version
from .budget import Budget
from .evalset import EvalSet, Example
from .exceptions import (
    ArtifactError,
    BudgetExceeded,
    ConfigError,
    DegenerateEvalError,
    LeakageDetected,
    RecensionError,
)
from .models import Message, MockModel, Model
from .objective import F1, ExactMatch, LLMJudge, Objective
from .optimizer import ReflectiveOptimizer
from .record import CandidateRecord, RoundRecord, RunRecord

__version__ = "0.1.0"

__all__ = [
    "ArtifactError",
    "Budget",
    "BudgetExceeded",
    "CandidateRecord",
    "ConfigError",
    "DegenerateEvalError",
    "EvalSet",
    "ExactMatch",
    "Example",
    "F1",
    "LLMJudge",
    "LeakageDetected",
    "Message",
    "MockModel",
    "Model",
    "Objective",
    "Provenance",
    "RecensionError",
    "ReflectiveOptimizer",
    "RejectedCandidate",
    "RoundRecord",
    "RunRecord",
    "TextArtifact",
    "Version",
    "__version__",
]
