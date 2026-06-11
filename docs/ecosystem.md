# Bring your own optimizer

recension is a measurement-and-governance shell, not an optimizer. The mechanics of generating good
candidate edits are well covered by projects like [DSPy](https://github.com/stanfordnlp/dspy) and
[GEPA](https://github.com/gepa-ai/gepa). recension's contribution is everything around the edit:
versioned artifacts with provenance, a held-out objective with a locked test split, significance and
guard gates, leakage detection, a cost ledger, and a tamper-evident audit record.

Those two concerns compose. The [`Proposer`](api.md#recension.proposer.Proposer) protocol is the seam:
supply the candidate edits from any engine you like, and recension keeps owning the measurement and the
record. The artifact, evalset, and record abstractions do not change.

## The Proposer protocol

A proposer does two things: turn observed failures into a short diagnosis, and turn that diagnosis into
distinct candidate revisions.

```python
class Proposer(Protocol):
    def diagnose(self, model, artifact_text, failures, *, seed=None) -> str: ...
    def propose(self, model, artifact_text, diagnosis, n, *, seed=None) -> list[str]: ...
```

The built-in [`DefaultProposer`](api.md#recension.proposer.DefaultProposer) is the reflective
diagnose/propose loop recension ships with. To plug in your own engine, pass a custom proposer to the
optimizer.

## Wrapping an external engine

[`CallableProposer`](api.md#recension.proposer.CallableProposer) adapts plain functions, so wrapping an
external optimizer is a few lines. Your function returns the candidate texts; recension scores them on
the held-out split, runs the significance and guard gates, checks for leakage, and writes the audit
record.

```python
from recension import CallableProposer, ReflectiveOptimizer

def propose_with_my_engine(model, artifact_text, diagnosis, n, seed):
    # Call DSPy, GEPA, or any optimizer here and return up to n revised artifact strings.
    return my_engine.optimize(artifact_text, n=n)

optimizer = ReflectiveOptimizer(
    artifact=artifact,
    evalset=evalset,
    objective=objective,
    model=model,
    proposer=CallableProposer(propose_with_my_engine),
)
record = optimizer.run()  # recension owns measurement, leakage, and the record
```

For full control, implement the `Proposer` protocol directly as a class. Either way, the run produces
the same [`RunRecord`](api.md#recension.record.RunRecord): the governance guarantees hold no matter who
proposed the edits.
