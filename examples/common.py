"""Shared plumbing for the worked examples.

Every example runs end-to-end against the deterministic ``MockModel`` with no
API key, so anyone can reproduce it. Pass ``--real`` to run the same example
against the Anthropic backend instead (requires ``recension[anthropic]`` and
``ANTHROPIC_API_KEY`` in the environment).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from recension import RunRecord
from recension.models import Message, MockModel, Model

OUTPUT_DIR = Path(__file__).parent / "output"


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--real",
        action="store_true",
        help="run against the Anthropic API instead of the offline mock "
        "(requires ANTHROPIC_API_KEY)",
    )
    return parser.parse_args()


def get_model(real: bool, script: Callable[[list[Message]], str]) -> Model:
    """The offline scripted mock, or the real Anthropic backend with --real."""
    if not real:
        return MockModel(script=script)
    from recension.models.anthropic import AnthropicModel

    return AnthropicModel()


def report(record: RunRecord, name: str) -> Path:
    """Save the run record next to the examples and print the summary."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{name}.json"
    record.save(path)
    print(record.summary())
    print(f"\nrun record: {path}", file=sys.stderr)
    return path
