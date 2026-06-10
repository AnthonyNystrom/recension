# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-06-10

Initial release.

### Added

- `TextArtifact`: versioned text with content-addressed version ids, unified
  diffs (stdlib `difflib`), append-only rollback, and full `Provenance` on
  every accepted version (diagnosis, scores, rejected sibling candidates, diff).
- `EvalSet` / `Example` with an enforced train/validation split; loaders
  `from_records` and `from_jsonl`. Split-integrity violations raise
  `DegenerateEvalError`.
- Objectives: `ExactMatch`, token-level `F1`, and model-graded `LLMJudge`
  (flagged as model-graded in run records).
- `ReflectiveOptimizer`: the propose/test/accept loop, with failures diagnosed on
  train, distinct candidates generated and scored on validation, acceptance
  gated on `min_improvement` plus leakage checks.
- `Budget`: caller-controlled candidates per round, rounds, diagnosis depth,
  and a hard `max_model_calls` ceiling (`BudgetExceeded` carries the partial
  audit record).
- Leakage heuristics: verbatim validation spans and implausible
  validation-vs-train gains; surfaced as flags, or raised via strict mode.
- `RunRecord` / `RoundRecord` / `CandidateRecord`: a complete, serializable
  audit record of every run, with the full artifact embedded.
- Model layer: provider-agnostic `Model` protocol, deterministic `MockModel`
  (the entire test suite runs offline), and an optional Anthropic backend
  (`recension[anthropic]`; API key from the environment only).
- CLI: `recension run --config run.yaml`, `recension show`, `recension diff`.
- Three reproducible worked examples and a docs site (MkDocs + API reference)
  with pages regenerated from real offline runs.
