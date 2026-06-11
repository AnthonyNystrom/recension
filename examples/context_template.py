"""Context template: text-layer changes move the metric with the model fixed.

Real-world analogue: a RAG / retrieval-QA pipeline where you tune how retrieved
chunks are assembled into the prompt, with the retriever and model held fixed.

The artifact here is not a prompt but a *template* that assembles retrieved
context and a question into the final message. The model never changes; only
the assembly text does. The starting template buries the context after the
question and never tells the model to ground its answer, so answers drift.
The accepted revision puts context first with a grounding instruction, and
the token-level F1 against reference answers moves accordingly.

Acceptance requires a *statistically significant* gain (``accept_significant``),
a locked *test* split gives an unbiased final estimate, and the record reports
per-*topic* slice scores so a gain on one subject area cannot hide a loss on
another.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from common import get_model, parse_args, report

from recension import (
    F1,
    Budget,
    EvalSet,
    Example,
    ReflectiveOptimizer,
    RunRecord,
    TextArtifact,
)
from recension.models import Message

STARTING_TEMPLATE = """\
Question: {question}
You may find this background useful:
{context}
Answer:"""

GROUNDING_LINE = "Answer using only the facts in the context above."

RECORDS = [
    {"id": "t1", "split": "train", "input": "When did the tower open?", "topic": "architecture",
     "context": "The tower opened to the public in March 1889 after 26 months of work.",
     "expected": "the tower opened in march 1889"},
    {"id": "t2", "split": "train", "input": "Who designed the bridge?", "topic": "architecture",
     "context": "The bridge was designed by the engineer Othmar Ammann in 1927.",
     "expected": "the bridge was designed by othmar ammann"},
    {"id": "t3", "split": "train", "input": "How long is the tunnel?", "topic": "architecture",
     "context": "At 57 kilometres, the tunnel is the longest rail tunnel in the world.",
     "expected": "the tunnel is 57 kilometres long"},
    {"id": "v1", "split": "validation", "input": "When did the museum open?", "topic": "culture",
     "context": "The museum opened in 1793 during the French Revolution.",
     "expected": "the museum opened in 1793"},
    {"id": "v2", "split": "validation", "input": "Who built the observatory?", "topic": "science",
     "context": "The observatory was built by the astronomer Tycho Brahe.",
     "expected": "the observatory was built by tycho brahe"},
    {"id": "v3", "split": "validation", "input": "How tall is the statue?", "topic": "culture",
     "context": "Including its pedestal, the statue stands 93 metres tall.",
     "expected": "the statue is 93 metres tall"},
    # A locked test split, scored once at the end on questions never seen.
    {"id": "x1", "split": "test", "input": "Who painted the ceiling?", "topic": "culture",
     "context": "The ceiling was painted by the artist Michelangelo over four years.",
     "expected": "the ceiling was painted by michelangelo"},
    {"id": "x2", "split": "test", "input": "How deep is the lake?", "topic": "science",
     "context": "At its deepest point the lake reaches 1642 metres.",
     "expected": "the lake is 1642 metres deep"},
]

_ANSWERS = {r["input"]: r["expected"] for r in RECORDS}


def render(template_text: str, example: Example) -> list[Message]:
    """Assemble the final message from the template (this is the artifact's job)."""
    prompt = template_text.replace("{question}", example.input).replace(
        "{context}", str(example.metadata.get("context", ""))
    )
    return [{"role": "user", "content": prompt}]


def mock_script(messages: list[Message]) -> str:
    """Offline stand-in: grounded assembly yields grounded answers.

    When the assembled prompt instructs the model to use only the context,
    the answer matches the reference; otherwise the model answers from vague
    memory with partial token overlap, exactly the failure mode retrieval
    templates exist to fix.
    """
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next(m["content"] for m in messages if m["role"] == "user")
    if system.startswith("You analyze why"):
        return (
            "The template puts the context after the question and never instructs "
            "the model to ground its answer in it, so answers come from memory."
        )
    if system.startswith("You revise text artifacts"):
        index = int(re.search(r"Propose revision (\d+)", user).group(1))  # type: ignore[union-attr]
        variants = {
            1: "Question: {question}\nBackground: {context}\nKeep the answer short.\nAnswer:",
            2: f"Context:\n{{context}}\n\n{GROUNDING_LINE}\n\nQuestion: {{question}}\nAnswer:",
            3: "{question}\n{context}",
        }
        return f"<revised_artifact>\n{variants[index]}\n</revised_artifact>"
    question = next((q for q in _ANSWERS if q in user), None)
    if question is None:
        return "I cannot find the question."
    if GROUNDING_LINE in user:
        return _ANSWERS[question]
    # Ungrounded: a vague answer sharing only a couple of tokens with the reference.
    return "it is " + " ".join(_ANSWERS[question].split()[-1:])


def build_optimizer(
    real: bool = False, on_progress: Callable[[str], None] | None = None
) -> ReflectiveOptimizer:
    """Construct the optimizer for this example (shared by the CLI run and the web demo)."""
    return ReflectiveOptimizer(
        artifact=TextArtifact.from_text(STARTING_TEMPLATE, name="context-template"),
        evalset=EvalSet.from_records(RECORDS),
        objective=F1(),
        model=get_model(real, mock_script),
        budget=Budget(candidates_per_round=3, rounds=2, diagnosis_depth=2, max_model_calls=200),
        seed=11,
        accept_significant=True,   # accept only a statistically significant validation gain
        slice_by="topic",          # report per-topic baseline -> final scores
        render=render,
        on_progress=on_progress,
    )


def main(real: bool = False) -> RunRecord:
    return build_optimizer(real=real).run()


if __name__ == "__main__":
    args = parse_args(__doc__ or "")
    record = main(real=args.real)
    report(record, "context_template")
