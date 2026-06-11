"""Regenerate the rendered example pages from the runnable scripts.

Runs every example end-to-end against ``MockModel`` (no API key, fully
deterministic) and renders one markdown page per example into
``docs/examples/generated/``, where the MkDocs site picks them up. Because the
pages are produced from real runs of the real scripts, the site can never
drift from the code.

Usage: ``python examples/build_site.py`` (or ``make examples``).
"""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLES_DIR = Path(__file__).parent
GENERATED_DIR = EXAMPLES_DIR.parent / "docs" / "examples" / "generated"

sys.path.insert(0, str(EXAMPLES_DIR))

import classification  # noqa: E402
import context_template  # noqa: E402
import skill_file  # noqa: E402
from common import report  # noqa: E402

from recension import RunRecord  # noqa: E402

EXAMPLES = [
    ("classification", "Classification prompt", classification),
    ("context_template", "Context template", context_template),
    ("skill_file", "Skill / instruction file", skill_file),
]


def _checks(c: object) -> str:
    """Significance and guard detail for a candidate, for the round table."""
    bits: list[str] = []
    sig = getattr(c, "significance", None)
    if sig is not None:
        verdict = "significant" if sig.significant else "not significant"
        bits.append(f"{verdict} (gain {sig.mean_difference:+.3f})")
    for g in getattr(c, "guard_scores", ()):
        arrow = "regressed" if g.regressed else "ok"
        bits.append(f"{g.name} {g.incumbent_score:.2f}->{g.candidate_score:.2f} ({arrow})")
    return "; ".join(bits) if bits else "none"


def render_page(name: str, title: str, doc: str, record: RunRecord) -> str:
    lines = [
        f"# {title}",
        "",
        doc.strip(),
        "",
        f"Runnable script: [`examples/{name}.py`]"
        f"(https://github.com/AnthonyNystrom/recension/blob/main/examples/{name}.py): "
        f"reproduce offline with `python examples/{name}.py`, or against the Anthropic "
        f"API with `python examples/{name}.py --real`.",
        "",
        "## Starting artifact",
        "",
        "```text",
        record.restored_artifact().history()[0].text.rstrip("\n"),
        "```",
        "",
        f"Baseline validation score: **{record.baseline_score:.4f}**"
        + (" *(model-graded)*" if record.model_graded else ""),
        "",
    ]
    for r in record.rounds:
        lines += [
            f"## Round {r.round_index}",
            "",
            f"Train score {r.train_score:.4f}; failures analyzed: "
            f"{', '.join(f'`{i}`' for i in r.failure_example_ids) or 'none'}.",
            "",
            f"**Diagnosis:** {r.diagnosis}",
            "",
            "| candidate | validation score | leakage flags | checks | outcome |",
            "|---|---|---|---|---|",
        ]
        for c in r.candidates:
            score = "n/a" if c.validation_score is None else f"{c.validation_score:.4f}"
            flags = "; ".join(c.leakage_flags) if c.leakage_flags else "none"
            outcome = "**accepted**" if c.accepted else "rejected"
            lines.append(
                f"| `{c.candidate_id}` | {score} | {flags} | {_checks(c)} | {outcome} |"
            )
        lines.append("")
        accepted = next((c for c in r.candidates if c.accepted), None)
        if accepted is not None:
            lines += [
                f"Accepted as version `{r.accepted_version_id}`:",
                "",
                "```diff",
                accepted.diff.rstrip("\n"),
                "```",
                "",
            ]
        else:
            lines += ["No candidate beat the incumbent this round.", ""]
    lines += [
        "## Result",
        "",
        f"Validation score **{record.baseline_score:.4f} → {record.final_score:.4f}** "
        f"({record.stopped_reason}, {record.total_model_calls} model calls).",
        "",
    ]
    if record.final_test_score is not None:
        gap = record.test_validation_gap if record.test_validation_gap is not None else 0.0
        overfit = " **possible overfitting to validation**" if record.validation_overfit else ""
        lines += [
            f"Locked test split (scored once at the end): "
            f"**{record.final_test_score:.4f}** (validation/test gap {gap:.4f}{overfit}).",
            "",
        ]
    if record.slice_scores:
        lines += [
            "**Per-slice scores**",
            "",
            "| slice | n | baseline | final |",
            "|---|---|---|---|",
        ]
        for s in record.slice_scores:
            mark = " (regressed)" if s.regressed else ""
            lines.append(
                f"| {s.slice} | {s.n} | {s.baseline_score:.4f} | {s.final_score:.4f}{mark} |"
            )
        lines.append("")
    if record.total_input_tokens or record.total_output_tokens:
        lines += [
            f"Token ledger: **{record.total_input_tokens} in / "
            f"{record.total_output_tokens} out**.",
            "",
        ]
    integrity = "verified" if not record.verify() else "FAILED"
    lines += [
        f"Integrity: **{integrity}** (fingerprint `{record.fingerprint()[:16]}`). "
        f"The complete audit record for this page is regenerated on every build.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    for name, title, module in EXAMPLES:
        record = module.main(real=False)
        report(record, name)
        page = render_page(name, title, module.__doc__ or "", record)
        path = GENERATED_DIR / f"{name}.md"
        path.write_text(page, encoding="utf-8")
        print(f"rendered {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
