# Worked examples

Three end-to-end optimization runs. Each consists of a runnable Python script in [`examples/`](https://github.com/AnthonyNystrom/recension/tree/main/examples) and a rendered page showing the actual run: the starting artifact, the diagnosis, the candidates compared, the accepted version with its diff, and the score progression.

Every example runs against the deterministic `MockModel` with **no API key**, so each page is reproducible by anyone:

```bash
python examples/classification.py
```

Pass `--real` to run the same script against the Anthropic API instead (requires `pip install "recension[anthropic]"` and `ANTHROPIC_API_KEY` in the environment).

Each example is a small, self-contained instance of one of the [use cases](../use-cases.md):

| Example | Use case | Shows |
|---|---|---|
| [Classification prompt](generated/classification.md) | Production classification / extraction (like a support-ticket triager) | The flagship loop: diagnose → propose → test → accept with an `ExactMatch` objective |
| [Context template](generated/context_template.md) | RAG context templates (like a retrieval QA pipeline) | Text-layer changes moving a token-level `F1` metric with the model held fixed |
| [Skill / instruction file](generated/skill_file.md) | Agent / skill instructions (like an agent system prompt) | A model-graded `LLMJudge` objective (flagged in the record) and a leakage flag on a cheating candidate |

Every run also produces a [`RunRecord`](../api.md#recension.record.RunRecord), the governance-and-audit artifact common to all four use cases.

The pages are regenerated from real runs with `make examples`, so the site cannot drift from the code.
