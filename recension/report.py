"""A self-contained HTML audit report rendered from a :class:`RunRecord`.

``render_report`` turns the full audit record into a single standalone HTML page
(inline CSS, no assets, no network) that a reviewer can open, share, or attach
to a change request. It surfaces everything the record carries: the baseline and
final scores, the locked test estimate and overfit flag, every round's diagnosis
and candidates (with significance, guard, and leakage detail), the accepted diff,
the per-slice breakdown, the token ledger, and the record's integrity status.
"""

from __future__ import annotations

import html

from .record import RoundRecord, RunRecord

__all__ = ["render_report"]

_STYLE = """\
:root { color-scheme: light dark; }
body { font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
       margin: 0; padding: 2rem; max-width: 60rem; margin-inline: auto; }
h1 { font-size: 1.5rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 .5rem; border-bottom: 1px solid #8884;
     padding-bottom: .25rem; }
.sub { color: #888; margin: 0 0 1rem; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(11rem, 1fr)); gap: .75rem;
         margin: 1rem 0; }
.stat { border: 1px solid #8884; border-radius: .5rem; padding: .6rem .8rem; }
.stat .k { color: #888; font-size: .8rem; text-transform: uppercase; letter-spacing: .03em; }
.stat .v { font-size: 1.2rem; font-weight: 600; }
.ok { color: #1a8a3b; } .bad { color: #c0392b; } .warn { color: #b8860b; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0; font-size: .92rem; }
th, td { text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #8883;
         vertical-align: top; }
th { color: #888; font-weight: 600; }
.badge { display: inline-block; padding: .05rem .45rem; border-radius: .35rem; font-size: .78rem;
         border: 1px solid #8886; }
.accepted { background: #1a8a3b22; border-color: #1a8a3b88; }
.rejected { color: #888; }
.round { border: 1px solid #8884; border-radius: .6rem; padding: .25rem 1rem 1rem; margin: 1rem 0; }
.diag { font-style: italic; color: #555; }
pre.diff { background: #8881; border-radius: .5rem; padding: .75rem; overflow-x: auto;
           font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; }
.diff .add { color: #1a8a3b; } .diff .del { color: #c0392b; } .diff .hunk { color: #6a5acd; }
"""


def render_report(record: RunRecord) -> str:
    """Render ``record`` as a complete, standalone HTML document (a string)."""
    name = html.escape(str(record.artifact.get("name", "artifact")))
    objective = html.escape(record.objective_name)
    graded = " (model-graded)" if record.model_graded else ""
    parts: list[str] = [
        "<!doctype html>",
        '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>recension audit: {name}</title>",
        f"<style>{_STYLE}</style></head><body>",
        f"<h1>recension audit: {name}</h1>",
        f'<p class="sub">objective: {objective}{graded} &middot; '
        f"stopped: {html.escape(record.stopped_reason)}</p>",
        _stats_block(record),
        _slices_block(record),
    ]
    parts.append("<h2>Rounds</h2>")
    if not record.rounds:
        parts.append("<p>No rounds were executed.</p>")
    for round_record in record.rounds:
        parts.append(_round_block(round_record))
    parts.append("</body></html>")
    return "\n".join(parts)


def _stat(key: str, value: str, cls: str = "") -> str:
    span = f'<span class="v {cls}">{value}</span>' if cls else f'<span class="v">{value}</span>'
    return f'<div class="stat"><div class="k">{html.escape(key)}</div>{span}</div>'


def _stats_block(record: RunRecord) -> str:
    cells = [
        _stat("baseline", f"{record.baseline_score:.4f}"),
        _stat("final (validation)", f"{record.final_score:.4f}"),
    ]
    if record.final_test_score is not None:
        cls = "warn" if record.validation_overfit else "ok"
        cells.append(_stat("test (locked, once)", f"{record.final_test_score:.4f}", cls))
        if record.validation_overfit:
            cells.append(_stat("overfit", "validation &gt; test", "warn"))
    cells.append(_stat("model calls", str(record.total_model_calls)))
    if record.total_input_tokens or record.total_output_tokens:
        cells.append(
            _stat("tokens", f"{record.total_input_tokens} in / {record.total_output_tokens} out")
        )
    problems = record.verify()
    if problems:
        cells.append(_stat("integrity", "FAILED", "bad"))
    else:
        cells.append(_stat("integrity", f"verified {record.fingerprint()[:12]}", "ok"))
    return f'<div class="stats">{"".join(cells)}</div>'


def _slices_block(record: RunRecord) -> str:
    if not record.slice_scores:
        return ""
    rows = ["<h2>Slices</h2>", "<table><tr><th>slice</th><th>n</th><th>baseline</th>"
            "<th>final</th><th></th></tr>"]
    for s in record.slice_scores:
        mark = '<span class="bad">regressed</span>' if s.regressed else ""
        rows.append(
            f"<tr><td>{html.escape(s.slice)}</td><td>{s.n}</td>"
            f"<td>{s.baseline_score:.4f}</td><td>{s.final_score:.4f}</td><td>{mark}</td></tr>"
        )
    rows.append("</table>")
    return "\n".join(rows)


def _round_block(r: RoundRecord) -> str:
    rows = [
        '<div class="round">',
        f"<h2>Round {r.round_index}</h2>",
        f'<p class="diag">diagnosis: {html.escape(r.diagnosis)}</p>',
        "<table><tr><th>candidate</th><th>score</th><th>status</th><th>detail</th></tr>",
    ]
    accepted_diff = ""
    for c in r.candidates:
        score = "n/a" if c.validation_score is None else f"{c.validation_score:.4f}"
        if c.accepted:
            status = '<span class="badge accepted">accepted</span>'
            accepted_diff = c.diff
        else:
            status = '<span class="rejected">rejected</span>'
        rows.append(
            f"<tr><td>{html.escape(c.candidate_id)}</td><td>{score}</td>"
            f"<td>{status}</td><td>{_candidate_detail(c)}</td></tr>"
        )
    rows.append("</table>")
    if accepted_diff:
        rows.append("<pre class=\"diff\">" + _render_diff(accepted_diff) + "</pre>")
    rows.append("</div>")
    return "\n".join(rows)


def _candidate_detail(c: object) -> str:
    bits: list[str] = []
    flags = getattr(c, "leakage_flags", ())
    if flags:
        bits.append('<span class="bad">flags: ' + html.escape(", ".join(flags)) + "</span>")
    sig = getattr(c, "significance", None)
    if sig is not None:
        verdict = "significant" if sig.significant else "not significant"
        cls = "ok" if sig.significant else "warn"
        bits.append(
            f'<span class="{cls}">{verdict}: gain {sig.mean_difference:+.4f}, '
            f"CI [{sig.ci_low:+.4f}, {sig.ci_high:+.4f}]</span>"
        )
    for g in getattr(c, "guard_scores", ()):
        cls = "bad" if g.regressed else "ok"
        bits.append(
            f'<span class="{cls}">{html.escape(g.name)}: '
            f"{g.incumbent_score:.4f}&rarr;{g.candidate_score:.4f}</span>"
        )
    return "<br>".join(bits)


def _render_diff(diff: str) -> str:
    out: list[str] = []
    for line in diff.splitlines():
        escaped = html.escape(line)
        if line.startswith("@@"):
            out.append(f'<span class="hunk">{escaped}</span>')
        elif line.startswith(("---", "+++")):
            out.append(escaped)  # file headers, not added/removed content
        elif line.startswith("+"):
            out.append(f'<span class="add">{escaped}</span>')
        elif line.startswith("-"):
            out.append(f'<span class="del">{escaped}</span>')
        else:
            out.append(escaped)
    return "\n".join(out)
