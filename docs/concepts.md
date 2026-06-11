# Concepts

This page walks the pipeline in the order a run actually happens, so the abstractions click into place
before you reach the [API reference](api.md). If you have read the [landing page](index.md), this is
the next step down in detail.

A run takes four things you supply (an artifact, an eval set, an objective, and a model) plus a budget,
and produces one thing: an audit record. Here is each piece in turn.

## The artifact: versioned text with provenance

A [`TextArtifact`](api.md#recension.artifact.TextArtifact) holds the text you are optimizing (a prompt,
a context template, an instruction file) together with an append-only history. Each accepted edit
appends a new version; nothing is rewritten in place, and even a rollback is recorded as a new version
rather than a pointer move, so the history of what was tried and reverted survives.

Every version after the first carries a [`Provenance`](api.md#recension.artifact.Provenance): the
diagnosis that motivated the change, the scores that justified it, the sibling candidates that lost,
and a unified diff against its parent. The diff is computed by the library, never supplied by the
caller, so it cannot drift from the actual text. Version ids are content-addressed, which is part of
what makes a seeded run reproducible.

## The eval set: a train/validation split that is enforced

An [`EvalSet`](api.md#recension.evalset.EvalSet) is a collection of
[`Example`](api.md#recension.evalset.Example) objects partitioned into `train` and `validation`, with
an optional locked `test` split. This partition is the integrity backbone of the whole library: the
optimizer diagnoses failures on `train` and decides acceptance only on `validation`. If the same
example could land in two splits, an edit could "improve" by memorizing a case it was also graded on,
so construction raises [`DegenerateEvalError`](api.md#recension.exceptions.DegenerateEvalError) on an
empty `train`/`validation`, a duplicate id, or any id shared across splits. Load from JSONL with
`EvalSet.from_jsonl`; a malformed line fails loudly with its file and line number rather than a bare
parser error.

### The locked test split, and why it matters

Acceptance is decided on `validation`, round after round, candidate after candidate. That repeated
selection is a multiple-comparisons problem: the final validation score is *optimistically biased*
upward simply because you kept the artifact that happened to score best on that particular set. Add a
**`test` split** (records with `split: "test"`) and the optimizer never touches it during the run; it
scores the final incumbent on it **exactly once** at the end. The record carries `final_test_score`
and the `validation`/`test` gap, and flags `validation_overfit` when the gap exceeds `overfit_gap`, so
an over-optimistic run announces itself rather than hiding.

## The objective: how an output is scored

An [`Objective`](api.md#recension.objective.Objective) maps `(model_output, example)` to a number where
higher is better, and aggregates per-example scores (mean by default). Three ship:

- [`ExactMatch`](api.md#recension.objective.ExactMatch) for label-style tasks.
- [`F1`](api.md#recension.objective.F1), token-level, for short free-form answers.
- [`LLMJudge`](api.md#recension.objective.LLMJudge), which scores against a rubric using a model when
  there is no gold answer. A judged run is marked **model-graded** in the record, so a reviewer always
  knows the acceptance metric was not reference-based.

## The loop: diagnose, propose, test, accept

[`ReflectiveOptimizer.run()`](api.md#recension.optimizer.ReflectiveOptimizer) scores the incumbent as a
baseline, then for each round:

1. **Diagnose.** Score the incumbent on `train`, take the worst examples (up to the budget's diagnosis
   depth), and ask the model for a structured hypothesis about what in the text caused them.
2. **Propose.** Generate several *distinct* candidate edits addressing the diagnosis. Near-duplicates
   are rejected and regenerated, because the value is in comparing genuinely different hypotheses, not
   rewordings.
3. **Test.** Score every candidate on `validation`.
4. **Accept.** The best candidate becomes the new incumbent only if it beats the current score by more
   than `min_improvement` and passes the leakage checks. Otherwise every candidate is recorded as
   rejected with its score, and the run stops or continues per policy.

The model is held fixed the entire time. Only the text changes, which is what lets you attribute a
metric move to the text layer rather than to the model.

### Significance, not just an epsilon

On a small validation set, a candidate can clear `min_improvement` purely by noise (one example flips).
Turn on `accept_significant` and acceptance additionally requires the gain to be **statistically
significant**: a seeded [paired bootstrap](api.md#recension.stats) on the per-example
`candidate - incumbent` differences must produce a confidence interval (at `alpha`) that excludes 0.
The bootstrap is recorded on the round's best candidate (effect, interval, verdict), so a reviewer
sees not just the score delta but whether it cleared the bar. It is off by default; when on, a
candidate that "won" by a single noisy example is rejected with its non-significant interval on the
record.

## Beyond one number: slices, guards, and cost

A single aggregate score hides as much as it shows. Three optional dimensions surface what it averages
away, all recorded for the reviewer:

- **Slices.** Set `slice_by` to an `Example.metadata` key and the record reports per-subgroup
  baseline-vs-final scores (a [`SliceScore`](api.md#recension.record.SliceScore) per value), so a run
  that lifts the overall number while regressing the `billing` segment is visible rather than averaged
  into the mean.
- **Guards.** Pass `guards=[...]` of secondary objectives that must not regress. A candidate that
  improves the primary metric but lowers a guard (for example
  [`MaxLength`](api.md#recension.objective.MaxLength), guarding output length) beyond `guard_tolerance`
  is rejected, with the incumbent-vs-candidate guard scores on the record. Guards are scored from the
  same model outputs, so reference-free guards add no model calls.
- **Cost.** Models that report token usage (a [`SupportsUsage`](api.md#recension.models.SupportsUsage)
  capability; `MockModel` reports synthetic counts, `AnthropicModel` reads `response.usage`) feed a
  token ledger. The record carries per-round and total input/output tokens, so a gain can be weighed
  against what it cost. Models without usage simply report zeros.

## The budget: update-time compute as a dial

A [`Budget`](api.md#recension.budget.Budget) controls candidates per round, number of rounds, diagnosis
depth, and a hard `max_model_calls` ceiling that counts task, diagnosis, proposal, and judge calls
alike. Hitting the ceiling raises [`BudgetExceeded`](api.md#recension.exceptions.BudgetExceeded) with
the partial audit record attached, so an interrupted run still leaves a trail. Nothing about
update-time compute is hardcoded in the loop.

## Leakage detection: honest heuristics, surfaced not hidden

Before an edit is committed, [`recension.leakage`](api.md#recension.leakage) runs two checks: it flags
a candidate that embeds a long verbatim span from a validation example, and one whose validation gain
is implausibly large relative to its train gain. These are heuristics, documented as such, not proofs.
By default a flag is **surfaced** in the record and the edit is still accepted with the flag attached;
strict mode turns a flag into [`LeakageDetected`](api.md#recension.exceptions.LeakageDetected) instead.
Either way, the signal is never silently dropped.

## The run record: the audit artifact

The output of a run is a [`RunRecord`](api.md#recension.record.RunRecord), fully serializable to JSON.
It captures the baseline, every round, every candidate (accepted and rejected) with its scores and
leakage flags, the diagnoses, the model-call counts, and timing, plus the full artifact with its
version history embedded. The bar it is built to: a reviewer who did not run the optimization can
reconstruct every decision from the record alone. Read one with `record.summary()` in code or
`recension show run_record.json` on the command line.

## Governance: gate, verify, share

The record is built to be acted on, not just filed:

- **Gate a prompt in CI.** `recension check --config run.yaml --baseline <record-or-score>` scores the
  current artifact and exits non-zero if it regressed below a baseline, so a prompt change that hurts
  your eval set fails the build like a failing test. See the [CLI guide](cli.md) for the GitHub Actions
  recipe.
- **Tamper-evidence.** Because artifact version ids are content-addressed,
  [`RunRecord.verify()`](api.md#recension.record.RunRecord) detects an edited version's text or id with
  no external reference; `recension verify` and the integrity line in `recension show` surface it.
  `fingerprint()` is a deterministic content hash, and records can be HMAC-signed for full-field
  tamper-evidence.
- **Shareable audits.** `recension report run_record.json -o report.html` renders the whole record into
  a single standalone HTML page (no assets, no network) to attach to a change request or review.

## Bring your own optimizer

recension owns the measurement-and-governance shell, not the optimizer mechanics. The
[`Proposer`](api.md#recension.proposer.Proposer) protocol is the seam: inject a custom proposer (for
example one wrapping DSPy or GEPA) with `ReflectiveOptimizer(proposer=...)` and recension keeps owning
versioning, held-out measurement, leakage, and the audit record while your engine supplies the edits.
See [Bring your own optimizer](ecosystem.md).

---

Next: see these pieces on real problems in [Use cases](use-cases.md), or watch a full run in the
[worked examples](examples/index.md).
