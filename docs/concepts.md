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
[`Example`](api.md#recension.evalset.Example) objects partitioned into `train` and `validation`. This
split is the integrity backbone of the whole library: the optimizer diagnoses failures on `train` and
decides acceptance only on `validation`. If the same example could land in both splits, an edit could
"improve" by memorizing a case it was also graded on, so construction raises
[`DegenerateEvalError`](api.md#recension.exceptions.DegenerateEvalError) on an empty split, a duplicate
id, or any id that appears in both. Load from JSONL with `EvalSet.from_jsonl`; a malformed line fails
loudly with its file and line number rather than a bare parser error.

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

---

Next: see these pieces on real problems in [Use cases](use-cases.md), or watch a full run in the
[worked examples](examples/index.md).
