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
from .objective import F1, ExactMatch, LLMJudge, MaxLength, Objective
from .optimizer import ReflectiveOptimizer, score_artifact
from .proposer import CallableProposer, DefaultProposer, FailureCase, Proposer
from .record import (
    CandidateRecord,
    GuardScore,
    RoundRecord,
    RunRecord,
    SignificanceRecord,
    SliceScore,
)
from .report import render_report

__version__ = "0.6.0"

__all__ = [
    "ArtifactError",
    "Budget",
    "BudgetExceeded",
    "CallableProposer",
    "CandidateRecord",
    "ConfigError",
    "DefaultProposer",
    "DegenerateEvalError",
    "EvalSet",
    "ExactMatch",
    "Example",
    "F1",
    "FailureCase",
    "GuardScore",
    "LLMJudge",
    "MaxLength",
    "LeakageDetected",
    "Message",
    "MockModel",
    "Model",
    "Objective",
    "Proposer",
    "Provenance",
    "RecensionError",
    "ReflectiveOptimizer",
    "RejectedCandidate",
    "RoundRecord",
    "RunRecord",
    "SignificanceRecord",
    "SliceScore",
    "TextArtifact",
    "render_report",
    "score_artifact",
    "Version",
    "__version__",
]
