# Command-line interface

The `recension` CLI is a thin wrapper over the library: `run` an optimization from a config file,
`show` a run record, and `diff` two artifact versions. Install the package and the `recension`
command is on your path.

## recension run

```bash
recension run --config run.yaml
```

Executes the optimization described by the config and writes the audit record to disk. A fully
commented config and a tiny dataset live in
[`examples/cli/`](https://github.com/AnthonyNystrom/recension/tree/main/examples/cli), runnable as-is
from the repo root:

```bash
recension run --config examples/cli/run.yaml
recension show run_record.json
```

!!! note
    The offline `mock` backend is deterministic but not task-aware, so a config-driven mock run
    demonstrates the pipeline and the record format rather than a real improvement. The
    [worked examples](examples/index.md) use scripted mocks to show runs that genuinely accept edits.
    For real optimization through the CLI, use `backend: anthropic` (needs the `[anthropic]` extra and
    `ANTHROPIC_API_KEY` in the environment).

### Config schema

The config is a YAML mapping. Bad values fail loudly with a `config error` and a non-zero exit.

| Key | Required | Notes |
|---|---|---|
| `artifact.text` / `artifact.path` | one of | Inline text, or a file to load. `artifact.name` is optional. |
| `evalset.path` | yes | JSONL file; each record needs `id`, `input`, `split` (`train`/`validation`/optional `test`), plus `expected` (for exact_match/f1) or `rubric` (for llm_judge). |
| `objective.name` | yes | `exact_match` (opt. `case_sensitive`), `f1`, or `llm_judge` (opt. `rubric`). |
| `model.backend` | yes | `mock` (opt. `seed`) or `anthropic` (opt. `model`; key from the env only). |
| `budget` | no | `candidates_per_round` (4), `rounds` (3), `diagnosis_depth` (1), `max_model_calls` (unlimited). |
| `seed` | no | Makes the whole run reproducible against the mock. |
| `min_improvement` | no | Margin a candidate must beat the incumbent by (default `1e-6`). |
| `accept_significant` | no | Require the validation gain to be statistically significant (a bootstrap CI excluding 0), not just above the margin (default `false`). |
| `alpha` | no | Significance level for the bootstrap CI (default `0.05` = 95%). |
| `bootstrap_resamples` | no | Resamples for the significance bootstrap (default `2000`). |
| `overfit_gap` | no | Flag `validation_overfit` when the final validation score beats the locked `test` split by more than this (default `0.1`). |
| `slice_by` | no | An `Example` metadata key; reports per-subgroup baseline-vs-final scores so a run that improves overall but regresses a segment is visible. |
| `slice_tolerance` | no | A slice is announced as regressed when it drops by more than this (default `0.0`). |
| `guards` | no | A list of non-regression guard objectives, e.g. `- {name: max_length, max_chars: 200}`. A candidate that improves the primary metric but regresses a guard is rejected. |
| `guard_tolerance` | no | Allowed drop on a guard before it counts as a regression (default `0.0`). |
| `strict_leakage` | no | Raise on a leakage-flagged winner instead of accepting it (default `false`). |
| `output` | no | Where to write the run record (default `run_record.json`). |
| `write_artifact` | no | Also write the final artifact text to this path. |

A `test` split (records with `split: "test"`) is scored exactly once at the end, giving an unbiased
estimate that the repeated selection on `validation` cannot give. See [Concepts](concepts.md).

If the run hits `budget.max_model_calls` or a leakage flag in strict mode, the CLI still writes the
partial record and exits with status `2`, so the audit trail survives the failure.

## recension show

```bash
recension show run_record.json
```

Prints a human-readable summary of a run record: the baseline, each round's diagnosis, every
candidate with its score and any leakage flags, the accepted version with its diff, and the final
score progression.

## recension diff

```bash
recension diff run_record.json <version-a> <version-b>
```

Prints the unified diff between two artifact versions stored in the record (the version ids appear in
`recension show`). Useful for reviewing exactly what an accepted edit changed.

## recension check

A prompt regression test for CI. It scores the **current** artifact (from the config) and exits
non-zero if it dropped below a baseline, so a prompt change that regresses your eval set fails the
build the same way a failing unit test would.

```bash
recension check --config run.yaml --baseline run_record.json   # baseline = a prior record's score
recension check --config run.yaml --baseline 0.82 --split test # or a literal score, on the test split
```

| Flag | Notes |
|---|---|
| `--config` | The same config `run` uses (artifact, evalset, objective, model). |
| `--baseline` | A prior record JSON (uses its `final_score`, or `final_test_score` with `--split test`) **or** a literal number. |
| `--split` | `validation` (default) or `test`. |
| `--tolerance` | Allowed drop below the baseline before failing (default `0.0`). |

Exit codes: `0` no regression, `1` regressed (or a config error). Wire it into a workflow to gate
merges on "the prompt did not get worse":

```yaml
# .github/workflows/prompt-check.yml
name: Prompt check
on: [pull_request]
jobs:
  check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
      - run: pip install recension
      - run: recension check --config run.yaml --baseline baseline_record.json --split test
```

## recension verify

Checks a run record's integrity and exits non-zero if it was tampered with.

```bash
recension verify run_record.json
```

Because artifact version ids are content-addressed, editing a version's text or id after the fact is
detected with no external reference. `recension show` prints the same integrity line. For full-record
tamper-evidence (any field), records can be HMAC-signed: with `RECENSION_SIGNING_KEY` set in the
environment, pass `--signature <hex>` to also check the signature. Exit codes: `0` verified, `2`
integrity failure.

## recension report

Renders a run record into a single standalone HTML page (inline CSS, no assets, no network) that you
can open, share, or attach to a change request.

```bash
recension report run_record.json -o report.html
```

The page shows the baseline and final scores, the locked test estimate and any overfit flag, every
round's diagnosis and candidates (with significance, guard, and leakage detail), the accepted diff,
the per-slice breakdown, the token ledger, and the record's integrity status.
