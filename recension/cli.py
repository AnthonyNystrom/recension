"""The ``recension`` command-line interface.

A thin wrapper over the library; no logic lives only here. Three commands:

- ``recension run --config run.yaml`` executes an optimization and writes the
  run record (and optionally the optimized artifact text) to disk.
- ``recension show record.json`` prints a human-readable summary of a record.
- ``recension diff record.json vA vB`` prints the diff between two artifact
  versions stored in a record.

Exit codes: 0 success; 1 configuration or usage error; 2 measurement-integrity
failure (budget exceeded or leakage in strict mode, where the partial record is
still written so the audit trail survives).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .artifact import TextArtifact
from .budget import Budget
from .evalset import EvalSet
from .exceptions import BudgetExceeded, ConfigError, LeakageDetected, RecensionError
from .models.base import Model
from .models.mock import MockModel
from .objective import F1, ExactMatch, LLMJudge, Objective
from .optimizer import ReflectiveOptimizer
from .record import RunRecord

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``recension`` console script."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result: int = args.func(args)
        return result
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1
    except RecensionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recension",
        description="Measured optimization of the text layer around a language model.",
    )
    sub = parser.add_subparsers(required=True)

    run = sub.add_parser("run", help="execute an optimization from a YAML config")
    run.add_argument("--config", required=True, help="path to the run config (YAML)")
    run.set_defaults(func=_cmd_run)

    show = sub.add_parser("show", help="print a human-readable summary of a run record")
    show.add_argument("record", help="path to a run record JSON file")
    show.set_defaults(func=_cmd_show)

    diff = sub.add_parser("diff", help="diff two artifact versions stored in a run record")
    diff.add_argument("record", help="path to a run record JSON file")
    diff.add_argument("version_a", help="version id (older)")
    diff.add_argument("version_b", help="version id (newer)")
    diff.set_defaults(func=_cmd_diff)
    return parser


# -- run ---------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    artifact = _build_artifact(_section(config, "artifact"))
    evalset = EvalSet.from_jsonl(_require(_section(config, "evalset"), "path"))
    model = _build_model(_section(config, "model"))
    objective = _build_objective(_section(config, "objective"), model)
    budget = _build_budget(config.get("budget", {}))
    output = config.get("output", "run_record.json")

    seed = config.get("seed")
    optimizer = ReflectiveOptimizer(
        artifact=artifact,
        evalset=evalset,
        objective=objective,
        model=model,
        budget=budget,
        seed=None if seed is None else _coerce(int, seed, "seed"),
        min_improvement=_coerce(float, config.get("min_improvement", 1e-6), "min_improvement"),
        strict_leakage=bool(config.get("strict_leakage", False)),
        on_progress=lambda line: print(line, file=sys.stderr),
    )
    try:
        record = optimizer.run()
    except (BudgetExceeded, LeakageDetected) as exc:
        # Fail loud, but never lose the audit trail.
        if exc.record is not None:
            exc.record.save(output)
            print(f"partial run record written to {output}", file=sys.stderr)
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    record.save(output)
    if "write_artifact" in config:
        Path(config["write_artifact"]).write_text(artifact.text, encoding="utf-8")
    print(f"run record written to {output}")
    print(
        f"validation score: {record.baseline_score:.4f} -> {record.final_score:.4f} "
        f"({record.stopped_reason}, {record.total_model_calls} model calls)"
    )
    return 0


def _load_config(path: str) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigError(f"config file not found: {path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError("config root must be a mapping")
    return loaded


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required config key: {key!r}")
    return mapping[key]


def _section(config: dict[str, Any], key: str) -> dict[str, Any]:
    """Return a required config section, erroring if it is not a mapping."""
    value = _require(config, key)
    if not isinstance(value, dict):
        raise ConfigError(f"config section {key!r} must be a mapping, got {type(value).__name__}")
    return value


def _coerce[T](fn: Callable[[Any], T], value: Any, key: str) -> T:
    """Coerce a config value, turning a bad value into a clean ConfigError."""
    try:
        return fn(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid value for {key!r}: {value!r} ({exc})") from exc


def _build_budget(section: Any) -> Budget:
    if not isinstance(section, dict):
        raise ConfigError("config section 'budget' must be a mapping")
    unknown = set(section) - {f.name for f in fields(Budget)}
    if unknown:
        raise ConfigError(f"unknown budget keys: {', '.join(sorted(unknown))}")
    try:
        return Budget.from_dict(section)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid budget: {exc}") from exc


def _build_artifact(section: dict[str, Any]) -> TextArtifact:
    if "path" in section:
        return TextArtifact.from_file(section["path"], name=section.get("name"))
    if "text" in section:
        return TextArtifact.from_text(section["text"], name=section.get("name", "artifact"))
    raise ConfigError("artifact section needs 'path' or 'text'")


def _build_model(section: dict[str, Any]) -> Model:
    backend = section.get("backend", "mock")
    if backend == "mock":
        return MockModel(seed=int(section.get("seed", 0)))
    if backend == "anthropic":
        # Imported lazily: requires the optional extra, and the API key comes
        # from the environment only; there is deliberately no key config field.
        from .models.anthropic import AnthropicModel

        kwargs: dict[str, Any] = {}
        if "model" in section:
            kwargs["model"] = section["model"]
        return AnthropicModel(**kwargs)
    raise ConfigError(f"unknown model backend {backend!r} (expected 'mock' or 'anthropic')")


def _build_objective(section: dict[str, Any], model: Model) -> Objective:
    name = _require(section, "name")
    if name == "exact_match":
        return ExactMatch(case_sensitive=bool(section.get("case_sensitive", False)))
    if name == "f1":
        return F1()
    if name == "llm_judge":
        return LLMJudge(model, rubric=section.get("rubric"))
    raise ConfigError(
        f"unknown objective {name!r} (expected 'exact_match', 'f1', or 'llm_judge')"
    )


# -- show / diff ---------------------------------------------------------------


def _cmd_show(args: argparse.Namespace) -> int:
    record = _load_record(args.record)
    print(record.summary())
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    record = _load_record(args.record)
    out = record.restored_artifact().diff(args.version_a, args.version_b)
    print(out if out else "(no differences)")
    return 0


def _load_record(path: str) -> RunRecord:
    record_path = Path(path)
    if not record_path.exists():
        raise ConfigError(f"record file not found: {path}")
    return RunRecord.load(record_path)
