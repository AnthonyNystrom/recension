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
| `evalset.path` | yes | JSONL file; each record needs `id`, `input`, `split` (`train`/`validation`), plus `expected` (for exact_match/f1) or `rubric` (for llm_judge). |
| `objective.name` | yes | `exact_match` (opt. `case_sensitive`), `f1`, or `llm_judge` (opt. `rubric`). |
| `model.backend` | yes | `mock` (opt. `seed`) or `anthropic` (opt. `model`; key from the env only). |
| `budget` | no | `candidates_per_round` (4), `rounds` (3), `diagnosis_depth` (1), `max_model_calls` (unlimited). |
| `seed` | no | Makes the whole run reproducible against the mock. |
| `min_improvement` | no | Margin a candidate must beat the incumbent by (default `1e-6`). |
| `strict_leakage` | no | Raise on a leakage-flagged winner instead of accepting it (default `false`). |
| `output` | no | Where to write the run record (default `run_record.json`). |
| `write_artifact` | no | Also write the final artifact text to this path. |

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
