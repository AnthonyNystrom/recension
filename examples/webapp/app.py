"""A small Flask app to watch ``recension`` optimize a prompt, live.

This is a demonstration tool, not part of the library. It runs the three
worked examples end-to-end against the deterministic ``MockModel`` (offline, no
API key) and streams the optimizer's propose/test/accept rounds to the browser
via Server-Sent Events, then renders the full audit record.

Run it::

    pip install "recension[webapp]"   # or: pip install flask
    python examples/webapp/app.py     # serves http://127.0.0.1:5000

The library is untouched: the app only imports each example's ``build_optimizer``
and feeds the optimizer's ``on_progress`` callback into the event stream.
"""

from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from flask import Flask, Response, render_template, stream_with_context

# The example modules live one directory up and import `from common import ...`,
# so put that directory on the path before importing them.
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import classification  # noqa: E402
import context_template  # noqa: E402
import skill_file  # noqa: E402

app = Flask(__name__)

# name -> (title, module, one-line blurb). Order is the display order.
EXAMPLES: dict[str, tuple[str, Any, str]] = {
    "classification": (
        "Classification prompt",
        classification,
        "Like a support-ticket triager: optimize a labeling prompt against a "
        "held-out set with an ExactMatch objective. The flagship example.",
    ),
    "context_template": (
        "Context template",
        context_template,
        "Like a retrieval QA pipeline: optimize how retrieved context is assembled "
        "into the prompt. The model is held fixed, so token-level F1 moves with the text.",
    ),
    "skill_file": (
        "Skill / instruction file",
        skill_file,
        "Like an agent system prompt: optimize a longer instruction file judged by "
        "an LLM rubric. Watch a leakage flag fire on a candidate that memorizes the set.",
    ),
}

# Presentational only: the offline mock runs in milliseconds, so pace the
# progress lines slightly to make the live stream legible. Lives here, not in
# the library.
_STEP_DELAY_SECONDS = 0.12


@app.route("/")
def index() -> str:
    """Landing page: the propose/collate/commit idea plus a card per example."""
    return render_template("index.html", examples=EXAMPLES)


@app.route("/favicon.ico")
def favicon() -> Response:
    """No favicon for the demo; answer the browser's automatic request quietly."""
    return Response(status=204)


@app.route("/run/<name>")
def run(name: str) -> Response:
    """Stream one example's optimization as Server-Sent Events.

    Emits ``log`` events (one per optimizer progress line), then a single
    ``record`` event carrying the complete run record as JSON, then ``done``.
    """
    if name not in EXAMPLES:
        return Response("unknown example", status=404)
    _, module, _ = EXAMPLES[name]
    events: queue.Queue[tuple[str, str]] = queue.Queue()

    def on_progress(line: str) -> None:
        time.sleep(_STEP_DELAY_SECONDS)
        events.put(("log", json.dumps(line)))

    def worker() -> None:
        try:
            record = module.build_optimizer(real=False, on_progress=on_progress).run()
            # fingerprint() and verify() are methods, not serialized fields, so
            # inject them for the page to show the integrity line.
            data = json.loads(record.to_json(indent=None))
            data["_fingerprint"] = record.fingerprint()
            data["_integrity_ok"] = not record.verify()
            events.put(("record", json.dumps(data)))
        except Exception as exc:  # surface failures to the page, never hide them
            events.put(("error", json.dumps(f"{type(exc).__name__}: {exc}")))
        finally:
            events.put(("done", "{}"))

    def stream() -> Any:
        threading.Thread(target=worker, daemon=True).start()
        while True:
            kind, payload = events.get()
            yield f"event: {kind}\ndata: {payload}\n\n"
            if kind == "done":
                break

    return Response(
        stream_with_context(stream()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    # DESIGN NOTE: default to 5001, not 5000. On macOS, port 5000 is taken by
    # the AirPlay Receiver (Control Center), which intercepts browser requests
    # to localhost:5000 even when this app is bound there. Override with
    # RECENSION_DEMO_PORT if 5001 is also busy.
    import os

    port = int(os.environ.get("RECENSION_DEMO_PORT", "5001"))
    print(f"recension demo: http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
