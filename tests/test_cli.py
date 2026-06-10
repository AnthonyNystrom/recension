"""CLI tests: run, show, diff — all offline against the mock backend."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from recension import RunRecord
from recension.cli import main


def write_inputs(tmp_path: Path, config_extra: dict[str, object] | None = None) -> Path:
    (tmp_path / "prompt.txt").write_text("Label the sentiment.", encoding="utf-8")
    rows = [
        {"id": "t1", "input": "a", "expected": "positive", "split": "train"},
        {"id": "t2", "input": "b", "expected": "negative", "split": "train"},
        {"id": "v1", "input": "c", "expected": "positive", "split": "validation"},
        {"id": "v2", "input": "d", "expected": "negative", "split": "validation"},
    ]
    (tmp_path / "examples.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    config: dict[str, object] = {
        "artifact": {"path": str(tmp_path / "prompt.txt")},
        "evalset": {"path": str(tmp_path / "examples.jsonl")},
        "objective": {"name": "exact_match"},
        "model": {"backend": "mock"},
        "budget": {"candidates_per_round": 2, "rounds": 1},
        "seed": 7,
        "output": str(tmp_path / "record.json"),
    }
    config.update(config_extra or {})
    config_path = tmp_path / "run.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


def test_run_writes_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_inputs(tmp_path)
    assert main(["run", "--config", str(config_path)]) == 0
    out = capsys.readouterr().out
    assert "run record written" in out
    record = RunRecord.load(tmp_path / "record.json")
    assert record.objective_name == "exact_match"
    assert record.seed == 7


def test_run_writes_optimized_artifact_when_asked(tmp_path: Path) -> None:
    target = tmp_path / "optimized.txt"
    config_path = write_inputs(tmp_path, {"write_artifact": str(target)})
    assert main(["run", "--config", str(config_path)]) == 0
    assert target.exists()


def test_run_budget_overrun_exits_2_and_saves_partial_record(tmp_path: Path) -> None:
    config_path = write_inputs(tmp_path, {"budget": {"max_model_calls": 1}})
    assert main(["run", "--config", str(config_path)]) == 2
    record = RunRecord.load(tmp_path / "record.json")
    assert record.stopped_reason == "budget_exceeded"


def test_show_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_inputs(tmp_path)
    main(["run", "--config", str(config_path)])
    capsys.readouterr()
    assert main(["show", str(tmp_path / "record.json")]) == 0
    out = capsys.readouterr().out
    assert "baseline" in out
    assert "stopped:" in out


def test_diff_prints_version_diff(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_inputs(tmp_path)
    main(["run", "--config", str(config_path)])
    record = RunRecord.load(tmp_path / "record.json")
    capsys.readouterr()
    version = record.baseline_version_id
    assert main(["diff", str(tmp_path / "record.json"), version, version]) == 0
    assert "no differences" in capsys.readouterr().out


def test_missing_config_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["run", "--config", str(tmp_path / "nope.yaml")]) == 1
    assert "config error" in capsys.readouterr().err


def test_unknown_objective_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_inputs(tmp_path, {"objective": {"name": "vibes"}})
    assert main(["run", "--config", str(config_path)]) == 1
    assert "unknown objective" in capsys.readouterr().err


def test_unknown_backend_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_inputs(tmp_path, {"model": {"backend": "gpt"}})
    assert main(["run", "--config", str(config_path)]) == 1
    assert "unknown model backend" in capsys.readouterr().err


def test_typoed_budget_key_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_inputs(tmp_path, {"budget": {"max_model_call": 5}})
    assert main(["run", "--config", str(config_path)]) == 1
    err = capsys.readouterr().err
    assert "config error" in err
    assert "max_model_call" in err


def test_non_numeric_seed_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_inputs(tmp_path, {"seed": "not-an-int"})
    assert main(["run", "--config", str(config_path)]) == 1
    assert "config error" in capsys.readouterr().err


def test_non_numeric_min_improvement_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_inputs(tmp_path, {"min_improvement": "lots"})
    assert main(["run", "--config", str(config_path)]) == 1
    assert "config error" in capsys.readouterr().err


def test_non_mapping_section_is_a_config_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_inputs(tmp_path, {"objective": "exact_match"})
    assert main(["run", "--config", str(config_path)]) == 1
    assert "must be a mapping" in capsys.readouterr().err


def test_missing_record_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["show", str(tmp_path / "missing.json")]) == 1
    assert "not found" in capsys.readouterr().err
