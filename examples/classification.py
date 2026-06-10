"""Classification prompt: the flagship example.

Real-world analogue: a production labeling or extraction prompt (support-ticket
triage, invoice field extraction, content moderation) tuned on labeled data.

A short sentiment-labeling prompt is optimized against a held-out set with an
``ExactMatch`` objective. The starting prompt never states the allowed labels,
so the model answers free-form and misses. The optimizer diagnoses that from
train failures, proposes four distinct revisions, and accepts the one that
states the labels, because it wins on the *validation* split, not because it
looked good to anyone.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from common import get_model, parse_args, report

from recension import Budget, EvalSet, ExactMatch, ReflectiveOptimizer, RunRecord, TextArtifact
from recension.models import Message

STARTING_PROMPT = "Decide how the customer feels about the product."

GOOD_INSTRUCTION = 'Answer with exactly one word: "positive" or "negative".'

RECORDS = [
    {"id": "t1", "input": "Absolutely love it, works perfectly", "expected": "positive",
     "split": "train"},
    {"id": "t2", "input": "Broke after two days, terrible build", "expected": "negative",
     "split": "train"},
    {"id": "t3", "input": "Great value and fast shipping", "expected": "positive",
     "split": "train"},
    {"id": "t4", "input": "Awful smell out of the box", "expected": "negative", "split": "train"},
    {"id": "v1", "input": "Terrible support experience", "expected": "negative",
     "split": "validation"},
    {"id": "v2", "input": "Love the design, great battery", "expected": "positive",
     "split": "validation"},
    {"id": "v3", "input": "Awful firmware, constant crashes", "expected": "negative",
     "split": "validation"},
    {"id": "v4", "input": "Great gift, my dad loves it", "expected": "positive",
     "split": "validation"},
]

_POSITIVE = ("love", "great", "perfect")
_NEGATIVE = ("terrible", "awful", "broke")


def mock_script(messages: list[Message]) -> str:
    """Offline stand-in for a frozen model.

    Task calls only produce a bare label once the artifact states the allowed
    labels; otherwise the model answers in a sentence, which ExactMatch
    rightly scores as wrong. Diagnosis and proposals are scripted to mirror
    what a real model does on this artifact.
    """
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next(m["content"] for m in messages if m["role"] == "user")
    if system.startswith("You analyze why"):
        return (
            "The artifact never tells the model the allowed labels or the output "
            "format, so it answers in free-form sentences that cannot exact-match."
        )
    if system.startswith("You revise text artifacts"):
        index = int(re.search(r"Propose revision (\d+)", user).group(1))  # type: ignore[union-attr]
        artifact = re.search(r"<artifact>\n(.*?)\n</artifact>", user, re.DOTALL).group(1)  # type: ignore[union-attr]
        variants = {
            1: "Read the customer message carefully and describe the overall sentiment.",
            2: f"{artifact}\n{GOOD_INSTRUCTION}",
            3: "Summarize the customer's feeling in a short phrase.",
            4: "Is this message happy or unhappy? Reply briefly.",
        }
        return f"<revised_artifact>\n{variants[index]}\n</revised_artifact>"
    sentiment = "positive" if any(w in user.lower() for w in _POSITIVE) else "negative"
    if GOOD_INSTRUCTION in system:
        return sentiment
    return f"The customer sounds {sentiment} about this."


def build_optimizer(
    real: bool = False, on_progress: Callable[[str], None] | None = None
) -> ReflectiveOptimizer:
    """Construct the optimizer for this example (shared by the CLI run and the web demo)."""
    return ReflectiveOptimizer(
        artifact=TextArtifact.from_text(STARTING_PROMPT, name="classification-prompt"),
        evalset=EvalSet.from_records(RECORDS),
        objective=ExactMatch(),
        model=get_model(real, mock_script),
        budget=Budget(candidates_per_round=4, rounds=2, diagnosis_depth=2, max_model_calls=200),
        seed=7,
        on_progress=on_progress,
    )


def main(real: bool = False) -> RunRecord:
    return build_optimizer(real=real).run()


if __name__ == "__main__":
    args = parse_args(__doc__ or "")
    record = main(real=args.real)
    report(record, "classification")
