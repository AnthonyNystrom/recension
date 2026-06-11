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
import os
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
from .objective import F1, ExactMatch, LLMJudge, MaxLength, Objective
from .optimizer import ReflectiveOptimizer, score_artifact
from .record import RunRecord
from .report import render_report

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

    check = sub.add_parser(
        "check", help="score the current artifact and fail if it regressed below a baseline"
    )
    check.add_argument("--config", required=True, help="path to the run config (YAML)")
    check.add_argument(
        "--baseline",
        required=True,
        help="a prior record JSON file, or a literal score, to compare against",
    )
    check.add_argument(
        "--split", choices=["validation", "test"], default="validation",
        help="which split to score on (default: validation)",
    )
    check.add_argument(
        "--tolerance", type=float, default=0.0,
        help="allowed drop below the baseline before failing (default: 0.0)",
    )
    check.set_defaults(func=_cmd_check)

    verify = sub.add_parser("verify", help="verify a run record's integrity")
    verify.add_argument("record", help="path to a run record JSON file")
    verify.add_argument(
        "--signature", help="expected HMAC signature (see RECENSION_SIGNING_KEY)"
    )
    verify.set_defaults(func=_cmd_verify)

    report = sub.add_parser(
        "report", help="render a standalone HTML audit report from a run record"
    )
    report.add_argument("record", help="path to a run record JSON file")
    report.add_argument(
        "-o", "--output", default="report.html",
        help="where to write the HTML (default: report.html)",
    )
    report.set_defaults(func=_cmd_report)
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
        overfit_gap=_coerce(float, config.get("overfit_gap", 0.1), "overfit_gap"),
        accept_significant=bool(config.get("accept_significant", False)),
        alpha=_coerce(float, config.get("alpha", 0.05), "alpha"),
        bootstrap_resamples=_coerce(
            int, config.get("bootstrap_resamples", 2000), "bootstrap_resamples"
        ),
        slice_by=config.get("slice_by"),
        slice_tolerance=_coerce(float, config.get("slice_tolerance", 0.0), "slice_tolerance"),
        guards=_build_guards(config.get("guards", [])),
        guard_tolerance=_coerce(float, config.get("guard_tolerance", 0.0), "guard_tolerance"),
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
    # Provider backends are imported lazily (each needs its optional extra), and
    # API keys come from the environment only; there is deliberately no key field.
    backend = section.get("backend", "mock")
    if backend == "mock":
        return MockModel(seed=int(section.get("seed", 0)))
    if backend == "anthropic":
        from .models.anthropic import AnthropicModel

        kwargs: dict[str, Any] = {}
        if "model" in section:
            kwargs["model"] = section["model"]
        return AnthropicModel(**kwargs)
    if backend == "openai":
        from .models.openai import OpenAIModel

        okwargs: dict[str, Any] = {}
        if "model" in section:
            okwargs["model"] = section["model"]
        if "base_url" in section:
            okwargs["base_url"] = section["base_url"]
        return OpenAIModel(**okwargs)
    if backend == "gemini":
        from .models.gemini import GeminiModel

        gkwargs: dict[str, Any] = {}
        if "model" in section:
            gkwargs["model"] = section["model"]
        return GeminiModel(**gkwargs)
    if backend == "ollama":
        from .models.ollama import OllamaModel

        lkwargs: dict[str, Any] = {}
        if "model" in section:
            lkwargs["model"] = section["model"]
        if "host" in section:
            lkwargs["host"] = section["host"]
        return OllamaModel(**lkwargs)
    raise ConfigError(
        f"unknown model backend {backend!r} "
        "(expected 'mock', 'anthropic', 'openai', 'gemini', or 'ollama')"
    )


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


def _build_guards(section: Any) -> list[Objective]:
    """Build guard objectives from the config (currently only ``max_length``)."""
    if not isinstance(section, list):
        raise ConfigError("config key 'guards' must be a list")
    guards: list[Objective] = []
    for i, spec in enumerate(section):
        if not isinstance(spec, dict) or "name" not in spec:
            raise ConfigError(f"guard {i} must be a mapping with a 'name'")
        name = spec["name"]
        if name == "max_length":
            guards.append(MaxLength(_coerce(int, _require(spec, "max_chars"), "max_chars")))
        else:
            raise ConfigError(f"unknown guard {name!r} (expected 'max_length')")
    return guards


# -- check (regression guard) --------------------------------------------------


def _cmd_check(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    artifact = _build_artifact(_section(config, "artifact"))
    evalset = EvalSet.from_jsonl(_require(_section(config, "evalset"), "path"))
    model = _build_model(_section(config, "model"))
    objective = _build_objective(_section(config, "objective"), model)
    examples = evalset.test if args.split == "test" else evalset.validation
    if not examples:
        raise ConfigError(f"the {args.split!r} split is empty; nothing to check against")

    baseline = _resolve_baseline(args.baseline, args.split)
    current = score_artifact(artifact.text, examples, objective, model)
    floor = baseline - args.tolerance
    print(
        f"{args.split} score: {current:.4f}  baseline: {baseline:.4f}  "
        f"floor: {floor:.4f} (tolerance {args.tolerance:.4f})"
    )
    if current < floor:
        print(
            f"REGRESSION: {current:.4f} is below the floor {floor:.4f}", file=sys.stderr
        )
        return 1
    print("OK: no regression")
    return 0


def _resolve_baseline(baseline: str, split: str) -> float:
    """A baseline is either a record file (use its final score) or a literal number."""
    path = Path(baseline)
    if path.exists():
        record = RunRecord.load(path)
        if split == "test":
            if record.final_test_score is None:
                raise ConfigError(
                    f"baseline record {baseline!r} has no test score to compare against"
                )
            return record.final_test_score
        return record.final_score
    try:
        return float(baseline)
    except ValueError as exc:
        raise ConfigError(
            f"--baseline {baseline!r} is neither a record file nor a number"
        ) from exc


# -- verify (integrity) --------------------------------------------------------


def _cmd_verify(args: argparse.Namespace) -> int:
    record = _load_record(args.record)
    problems = record.verify()
    if args.signature is not None:
        key = os.environ.get("RECENSION_SIGNING_KEY")
        if not key:
            raise ConfigError(
                "--signature given but RECENSION_SIGNING_KEY is not set in the environment"
            )
        if not record.verify_signature(key, args.signature):
            problems.append("HMAC signature does not match (record was altered or wrong key)")
    if problems:
        print("integrity: FAILED", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2
    print(f"integrity: verified  (fingerprint {record.fingerprint()[:16]})")
    return 0


# -- report (HTML audit) -------------------------------------------------------


def _cmd_report(args: argparse.Namespace) -> int:
    record = _load_record(args.record)
    Path(args.output).write_text(render_report(record), encoding="utf-8")
    print(f"audit report written to {args.output}")
    return 0


# -- show / diff ---------------------------------------------------------------


def _cmd_show(args: argparse.Namespace) -> int:
    record = _load_record(args.record)
    print(record.summary())
    problems = record.verify()
    if problems:
        print(f"integrity: FAILED ({len(problems)} problem(s); run `recension verify`)")
    else:
        print(f"integrity: verified (fingerprint {record.fingerprint()[:16]})")
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
