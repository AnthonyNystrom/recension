<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/recension-logo-dark.svg">
    <img src="docs/assets/recension-logo.svg" alt="recension" width="340">
  </picture>
</p>

# recension

Measured optimization of the text layer around a language model (prompts, context templates, skill and instruction files) with the rigor normally reserved for weight training: a held-out objective, a baseline, versioned artifacts, and a complete audit trail.

The name comes from textual criticism. A *recension* is the revision of a text by collating variant readings and keeping the best-supported one. That is the loop this library runs: propose multiple candidate edits, test each against held-out evidence, commit only what measurably improves, and record why.

## Why

The usual way to improve a prompt is to edit it, eyeball a few outputs, and ship. There is no held-out measurement, no record of why a change was made, and no defense against overfitting to the handful of cases you inspected. `recension` replaces that loop with a measured one:

- **No edit is accepted without a held-out score that beats the incumbent.** Failures are diagnosed on a train split; acceptance happens only on a validation split, and can require the gain to be *statistically significant* rather than above an epsilon. An optional locked test split gives an unbiased final estimate.
- **Every accepted version carries provenance**: the failures that motivated it, the diagnosis, every sibling candidate considered (with scores), and the diff. A reviewer who didn't run the optimization can reconstruct every decision.
- **One number doesn't hide regressions.** Optional per-slice scores, non-regression guard objectives, and a token-cost ledger surface what an aggregate averages away.
- **Leakage is checked, not assumed away.** Heuristics flag candidates that embed validation content or show implausible validation gains.
- **Records are built to be acted on.** Gate a prompt in CI with `recension check`, detect tampering with `recension verify`, and share a standalone HTML audit with `recension report`.
- **Compute is a dial.** Candidates per round, rounds, diagnosis depth, and a hard ceiling on model calls are all caller-controlled.

## Real-world use cases

- **Production classification and extraction** (support-ticket triage, invoice fields, moderation): improve a labeling prompt on labeled data with measured, regression-safe edits.
- **RAG context templates**: tune how retrieved chunks are assembled into the prompt with the model held fixed, so the metric move is attributable to the text.
- **Agent and skill instructions**: optimize longer instruction files judged by an `LLMJudge` rubric when there is no gold answer.
- **Governance and audit**: ship a replayable, tamper-evident `RunRecord` for every prompt change; gate merges in CI with `recension check`, and hand reviewers a standalone HTML audit with `recension report`.

Full write-ups, plus a "how it works" walkthrough, are on the [documentation site](https://anthonynystrom.github.io/recension/).

## Prior art, honestly

DSPy and GEPA own the optimization mechanics this library's `ReflectiveOptimizer` performs; if you want state-of-the-art prompt optimization algorithms, look there. `recension`'s contribution is the **measurement and governance shell** around a text artifact: versioned artifacts with provenance, leakage detection, the complete audit record, and budgeted update-time compute. That delegation is a real seam, not a promise: the `Proposer` protocol lets an external engine supply the candidate edits while recension keeps owning the artifact, the measurement, and the record (see the [Bring your own optimizer](https://anthonynystrom.github.io/recension/ecosystem/) guide).

## Install

```bash
pip install recension              # core: zero provider dependencies
pip install "recension[anthropic]" # Anthropic backend
pip install "recension[openai]"    # OpenAI (and OpenAI-compatible servers)
pip install "recension[gemini]"    # Google Gemini
# Ollama (local models) needs no extra: it uses the standard library
```

Python 3.12+. The core (and the whole test suite) runs against a deterministic `MockModel` with no API key and no network.

## Quickstart

```python
from recension import (
    Budget, EvalSet, ExactMatch, MockModel, ReflectiveOptimizer, TextArtifact,
)

artifact = TextArtifact.from_text("Label the sentiment of the message.")

# Held-out examples, split into train (for diagnosis) and validation (for
# acceptance). Load from a JSONL file with EvalSet.from_jsonl(path) instead.
evalset = EvalSet.from_records([
    {"id": "t1", "input": "Absolutely love this", "expected": "positive", "split": "train"},
    {"id": "t2", "input": "Broke after a day", "expected": "negative", "split": "train"},
    {"id": "v1", "input": "Terrible support", "expected": "negative", "split": "validation"},
    {"id": "v2", "input": "Exceeded expectations", "expected": "positive", "split": "validation"},
])

optimizer = ReflectiveOptimizer(
    artifact=artifact,
    evalset=evalset,
    objective=ExactMatch(),
    model=MockModel(),          # offline mock; see below for the real backend
    budget=Budget(candidates_per_round=4, rounds=3, max_model_calls=200),
    seed=7,
)
record = optimizer.run()

print(record.summary())         # baseline → accepted versions → final score
record.save("run_record.json")  # the complete audit artifact
```

To run against a real model, install the matching extra, set the provider's key in your environment, and pass the backend. The core never imports a provider, so each is kept off the top-level import:

```python
from recension.models.anthropic import AnthropicModel  # ANTHROPIC_API_KEY
from recension.models.openai import OpenAIModel         # OPENAI_API_KEY (also OpenAI-compatible servers via base_url)
from recension.models.gemini import GeminiModel         # GEMINI_API_KEY / GOOGLE_API_KEY
from recension.models.ollama import OllamaModel         # local Ollama server, no key

optimizer = ReflectiveOptimizer(..., model=OpenAIModel(model="gpt-4o-mini"))
```

Any object satisfying the small [`Model`](recension/models/base.py) protocol works, so bringing your own provider is a ~30-line adapter (`MockModel` is a worked template). API keys are read from the environment only, never from code or config. One honesty note: deterministic, reproducible runs are a guarantee of `MockModel` only; hosted APIs treat `seed` as a best-effort hint or ignore it.

## CLI

```bash
recension run --config run.yaml      # execute an optimization, write the record
recension show run_record.json       # baseline, accepted diffs, score progression, integrity
recension diff run_record.json vA vB # diff between two artifact versions
recension check --config run.yaml --baseline run_record.json  # CI guard: exit non-zero on regression
recension verify run_record.json     # detect tampering (content-addressed version chain)
recension report run_record.json -o report.html  # standalone HTML audit page
```

A runnable, fully commented config and dataset live in [`examples/cli/`](examples/cli) (`recension run --config examples/cli/run.yaml`); the config schema and a GitHub Actions recipe for `recension check` are in the [CLI guide](https://anthonynystrom.github.io/recension/cli/).

## Documentation

Full docs, API reference, and three worked examples (each reproducible offline against `MockModel`): **https://anthonynystrom.github.io/recension/**

## License

MIT
