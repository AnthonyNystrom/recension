# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
adheres to [Semantic Versioning](https://semver.org/).

## [0.6.1] - 2026-06-11

### Fixed

- README logo now uses absolute raw-GitHub PNG URLs instead of relative SVG paths, so it renders on
  the PyPI project page (PyPI does not resolve relative links or display SVG images). No code changes.

## [0.6.0] - 2026-06-11

More model providers. The core stays provider-agnostic; each hosted backend is an optional extra and
reads its key from the environment only.

### Added

- `OpenAIModel` (extra `openai`): the Chat Completions API, which also drives OpenAI-compatible servers
  (vLLM, LM Studio, OpenRouter, ...) via `base_url`. Reasoning models (o-series, gpt-5) are handled
  correctly: `max_completion_tokens` instead of `max_tokens`, and `temperature` dropped (with a
  `send_temperature` override).
- `GeminiModel` (extra `gemini`): the Google Gemini API (google-genai), mapping `system` messages to
  the system instruction and `assistant` to Gemini's `model` role.
- `OllamaModel` (no extra): local models via a running Ollama server's chat API, using only the
  standard library.
- All three report token usage into the cost ledger and accept an injected client/transport for
  testing. CLI `model.backend` now accepts `openai`, `gemini`, and `ollama` (with `base_url` / `host`).

## [0.5.0] - 2026-06-11

Shareable audits and an ecosystem seam.

### Added

- `recension report <record.json> -o report.html` renders a run record into a single standalone HTML
  audit page (inline CSS, no assets, no network): baseline and final scores, the locked test estimate
  and overfit flag, every round's diagnosis and candidates with significance/guard/leakage detail, the
  accepted diff, the per-slice breakdown, the token ledger, and the integrity status. New public
  `render_report`.
- A pluggable `Proposer` interface (`ReflectiveOptimizer(proposer=...)`) so candidate generation can be
  supplied by an external optimizer (DSPy, GEPA, or your own) while recension keeps owning versioning,
  held-out measurement, leakage detection, and the audit record. Ships `DefaultProposer` (the built-in
  reflective loop) and `CallableProposer` (wrap plain functions). New "Bring your own optimizer" docs.

## [0.4.0] - 2026-06-11

Multi-dimensional evaluation: one number hides regressions. All additions are opt-in.

### Added

- Per-slice reporting: set `slice_by` to an `Example.metadata` key and the record carries per-subgroup
  baseline-vs-final scores (`SliceScore`), so a run that improves overall while regressing a segment is
  visible. `slice_tolerance` controls when a slice is announced as regressed.
- Guarded acceptance: `guards=[...]` of secondary objectives that must not regress. A candidate that
  improves the primary metric but lowers a guard beyond `guard_tolerance` is rejected, with the
  incumbent-vs-candidate guard scores recorded (`GuardScore`). Ships the `MaxLength` guard objective.
- Cost ledger: an optional model-usage capability (`SupportsUsage` / `TokenUsage`). `MockModel` reports
  synthetic deterministic counts; `AnthropicModel` reads `response.usage`. The record carries per-round
  and total input/output tokens. Models without usage report zeros.
- CLI config keys: `slice_by`, `slice_tolerance`, `guards`, `guard_tolerance`.

## [0.3.0] - 2026-06-11

Prompts as tested, tamper-evident artifacts.

### Added

- `recension check`: a prompt regression guard for CI. Scores the current artifact against a baseline
  (a prior record's score or a literal) on the validation or test split and exits non-zero on a
  regression, so a prompt change that hurts your eval set fails the build. Backed by the new public
  `score_artifact` helper. Docs include a GitHub Actions recipe.
- Tamper-evident records: `RunRecord.verify()` checks the embedded artifact's content-addressed
  version chain (catching edited text/ids with no external reference), `RunRecord.fingerprint()` is a
  deterministic content hash, and `sign()` / `verify_signature()` add optional HMAC signing
  (`RECENSION_SIGNING_KEY`). New `recension verify` command; `recension show` prints an integrity line.
  `TextArtifact.verify()` exposes the version-chain check directly.

## [0.2.0] - 2026-06-10

Honest measurement. Both additions are opt-in; 0.1.0 code is unaffected.

### Added

- Optional locked `test` split on `EvalSet` (records with `split: "test"`). The optimizer scores the
  final incumbent on it exactly once and records `final_test_score`, the `validation`/`test` gap, and
  a `validation_overfit` flag when the gap exceeds `overfit_gap` (default 0.1). This gives an unbiased
  final estimate that the repeated selection on `validation` cannot.
- Significance-based acceptance (`accept_significant`, with `alpha` and `bootstrap_resamples`). When
  on, a candidate is accepted only if its validation gain is statistically significant (a seeded
  paired-bootstrap confidence interval excluding 0), not merely above `min_improvement`. The bootstrap
  is recorded on the round's best candidate. New stdlib-only `recension.stats` module; new
  `SignificanceRecord`.
- CLI config keys for the above (`accept_significant`, `alpha`, `bootstrap_resamples`, `overfit_gap`)
  and a `test` split in `examples/cli/`.

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
