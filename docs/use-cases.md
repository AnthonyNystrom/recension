# Use cases

`recension` is for the moment a prompt stops being a scratch experiment and becomes a **production
artifact**: something a real workload depends on, that a teammate might change next quarter, and that
someone may later have to explain. At that point "edit it and eyeball a few outputs" is not enough.
The four scenarios below are where teams reach for it. Each maps to a runnable [worked
example](examples/index.md) you can reproduce offline.

---

## 1. Production classification and extraction

**Scenario.** A support-ticket triager labels each incoming ticket `billing` / `bug` / `account`. An
invoice reader extracts the total and due date. A moderation prompt tags policy violations. You have a
few hundred labeled examples and a prompt that is "mostly right."

**Why the naive loop fails.** You inspect the twenty tickets that look wrong, tweak the prompt until
those twenty pass, and ship. But you never measured the other examples, so the tweak that fixed the
inspected twenty may have quietly regressed thirty others. And six months later, nobody remembers why
the label set grew a fourth category or why the output format changed.

**With recension.** Put the labeled data in an [`EvalSet`](api.md#recension.evalset.EvalSet) with a
train/validation split and score with [`ExactMatch`](api.md#recension.objective.ExactMatch) or
[`F1`](api.md#recension.objective.F1). The optimizer diagnoses failures on train and only accepts an
edit that beats the incumbent on the **held-out** validation split, so a fix that overfits the
inspected cases never gets committed. The flagship [Classification prompt](examples/generated/classification.md)
example walks the whole loop.

---

## 2. RAG context templates

**Scenario.** You answer questions over retrieved documents. The retriever and the model are fixed;
what you control is the **template** that assembles the retrieved chunks, the question, and the
instructions into the final prompt. Small wording and ordering choices move accuracy more than people
expect.

**Why the naive loop fails.** You change three things at once (a new template, a different chunk
count, a model upgrade), the metric moves, and you cannot say which change earned it. Next sprint the
metric drops and you are back to guessing.

**With recension.** Optimize only the template text while the **model is held frozen**, so any metric
move is attributable to the text layer, not the model or the retriever. Score the assembled answers
against reference QA pairs with [`F1`](api.md#recension.objective.F1) and supply a custom `render` that
splices retrieved context into the template. The [Context template](examples/generated/context_template.md)
example shows a grounding instruction earning the gain with the model fixed.

---

## 3. Agent and skill instructions

**Scenario.** You maintain a longer instruction file: an agent's system prompt, a "skill" that tells
the model how to write an executive summary, a style guide. There is no single gold answer, so
"better" is a matter of degree.

**Why the naive loop fails.** Without a gold label you fall back entirely on taste, which does not
scale past a handful of reviewed outputs and leaves no defensible record of why the instructions
changed. It is also where overfitting hides best: a candidate that pastes an evaluation example into
the instructions as a "worked example" can score beautifully and generalize terribly.

**With recension.** Score with [`LLMJudge`](api.md#recension.objective.LLMJudge) against an explicit
rubric. The run is flagged **model-graded** in the record so a reviewer knows the metric is not
reference-based, and the [leakage heuristics](api.md#recension.leakage) flag a candidate that embeds
verbatim validation content. The [Skill / instruction file](examples/generated/skill_file.md) example
shows exactly that flag firing on a cheating candidate.

---

## 4. Governance and audit

**Scenario.** You work somewhere a prompt change needs a paper trail: a regulated industry, a
review-heavy team, an incident retro that asks "why did the model start doing this?" The question is
not only "is the new prompt better" but "can we prove why we changed it, and that the improvement was
real."

**Why the naive loop fails.** A git diff of a prompt file tells you *what* changed, never *why*, on
what evidence, or what alternatives were rejected. And a number in a spreadsheet does not tell you
whether the gain was a genuine improvement or a memorized shortcut.

**With recension.** This is cross-cutting: every run, whatever the objective, produces a
[`RunRecord`](api.md#recension.record.RunRecord) complete enough that a reviewer who did not run the
optimization can reconstruct every decision: the failures that motivated the change, the diagnosis,
**every** candidate considered (accepted and rejected) with its scores, the leakage flags, and the
diff. Accepted versions carry that provenance on the artifact itself, the
[leakage checks](api.md#recension.leakage) stop overfit "wins" from shipping, and
[`Budget`](api.md#recension.budget.Budget) caps update-time spend. Inspect any run with
`recension show run_record.json`.

---

## When recension is not the right tool

It optimizes a text artifact against a held-out objective. If you do not have data to measure against,
there is nothing to optimize honestly, so start by building a small eval set. If what you want is a
better optimization *algorithm*, DSPy and GEPA own that ground; recension is the measurement and
governance shell around the artifact, not a competitor on optimizer mechanics. See
[Concepts](concepts.md) for how the pieces fit together.
