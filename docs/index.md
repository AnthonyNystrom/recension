# recension

**Measured optimization of the text layer around a language model** (prompts, context templates, skill and instruction files) with the rigor normally reserved for weight training: a held-out objective, a baseline, versioned artifacts, and a complete audit trail.

The name comes from textual criticism. A *recension* is the revision of a text by collating variant readings and keeping the best-supported one. That is the loop this library runs.

## Who it's for

Two readers, one tool:

- **The engineer shipping an LLM feature** who has started treating a prompt as a production artifact: something tested before it changes, protected from silently regressing, and improved on evidence rather than taste.
- **The team that has to justify a change** later: a reviewer, a regulator, an incident retro asking "why did the model start doing this?" Every accepted edit carries the evidence and reasoning that produced it.

If you have (or can build) a small set of labeled or judgeable examples and you care about not shipping an overfit "win," this is for you. See [Use cases](use-cases.md) for four concrete scenarios: production classification and extraction, RAG context templates, agent and skill instructions, and governance and audit.

## When to use it, and when not

Reach for `recension` when you have held-out data and you care about regression-safety and an audit trail for prompt changes. Reach for something else when you have no way to measure (build an eval set first), or when what you actually want is a better optimization *algorithm* (DSPy and GEPA own that; see [Prior art](#prior-art-honestly) below).

## The loop, in plain language

The usual way to improve a prompt is to edit it, eyeball a few outputs, and ship. Nothing measures whether the edit helped, nothing records why it was made, and nothing stops it from overfitting to the handful of cases you looked at.

`recension` replaces that loop with a measured one:

1. **Propose.** The optimizer diagnoses failures on a *train* split and generates several genuinely different candidate edits: different hypotheses about the fix, not rewordings.
2. **Collate.** Every candidate is scored on a held-out *validation* split. The model stays frozen the whole time; only the text changes, so the measured effect is the text's.
3. **Commit.** A candidate becomes the new version only if it beats the incumbent on validation by a margin, and only after leakage heuristics check that the gain wasn't bought by memorizing the held-out set. The new version carries full provenance: the failures that motivated it, the diagnosis, every sibling candidate with its score, and the diff.

The result of a run is a [`RunRecord`](api.md#recension.record.RunRecord), an audit artifact complete enough that a reviewer who didn't run the optimization can reconstruct every decision.

## Why measurement and provenance matter

- **No edit is accepted on vibes.** Acceptance happens on validation data the diagnosis never saw, and can require the gain to be [statistically significant](concepts.md#significance-not-just-an-epsilon), not just above an epsilon. An optional locked [test split](concepts.md#the-locked-test-split-and-why-it-matters) gives an unbiased final estimate.
- **Every decision is reviewable.** Accepted *and rejected* candidates are recorded with their scores, diffs, and leakage flags.
- **One number does not hide regressions.** Optional [per-slice scores, guard objectives, and a token-cost ledger](concepts.md#beyond-one-number-slices-guards-and-cost) surface what an aggregate averages away.
- **Integrity failures are loud.** Leakage, degenerate eval sets, and budget overruns raise or flag; they never pass silently. Even when a run fails, the partial audit record survives on the exception.
- **Records are built to be acted on.** [Gate a prompt in CI](concepts.md#governance-gate-verify-share) with `recension check`, detect tampering with `recension verify`, and share a standalone HTML audit with `recension report`.
- **Compute is a dial.** Candidates per round, rounds, diagnosis depth, and a hard model-call ceiling are all caller-controlled via [`Budget`](api.md#recension.budget.Budget).

## Install

```bash
pip install recension              # core: zero provider dependencies
pip install "recension[anthropic]" # adds the Anthropic backend
```

Python 3.12+. The core and the entire test suite run offline against a deterministic `MockModel`.

## Where to go next

- [Use cases](use-cases.md): four real-world scenarios, from support-ticket triage to compliance audit trails.
- [Concepts](concepts.md): how the pieces fit together, walked in run order.
- [CLI](cli.md): `run`, `show`, `diff`, plus `check` (CI guard), `verify` (integrity), and `report` (HTML audit).
- [Bring your own optimizer](ecosystem.md): plug DSPy, GEPA, or your own engine into the governance shell.
- [Worked examples](examples/index.md): three end-to-end runs, each reproducible offline with no API key.
- [API reference](api.md): generated from the docstrings.

## Prior art, honestly

DSPy and GEPA own the optimization mechanics this library's `ReflectiveOptimizer` performs; if you want state-of-the-art prompt-optimization algorithms, look there. `recension`'s contribution is the **measurement and governance shell** around a text artifact: versioned artifacts with provenance, leakage detection, the complete audit record, and budgeted update-time compute. That delegation is a real seam, not a promise: the [`Proposer`](api.md#recension.proposer.Proposer) protocol lets an external engine supply the candidate edits while recension keeps owning the artifact, the measurement, and the record. See [Bring your own optimizer](ecosystem.md).
