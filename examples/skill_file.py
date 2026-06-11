"""Skill / instruction file: model-graded objectives and leakage flags.

Real-world analogue: a longer instruction artifact (an agent's system prompt, a
reusable "skill" file, a style guide) optimized against a rubric when there is
no single gold answer to score against.

The artifact is a longer instruction file telling a model how to write
executive summaries. There is no gold answer to exact-match against, so an
``LLMJudge`` scores outputs against a rubric, and the run record flags the
whole run as *model-graded* so a reviewer knows the metric is not
reference-based.

One scripted candidate also pastes a validation example's text straight into
the instructions. It scores well, and the leakage heuristics flag it in the
record, which is exactly the point: a gain you got by memorizing the held-out
set is not a gain.

Acceptance requires a *statistically significant* gain (``accept_significant``),
a locked *test* split gives an unbiased final estimate, and a ``MaxLength``
*guard* confirms the structured rewrite also keeps summaries tight.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from common import get_model, parse_args, report

from recension import (
    Budget,
    EvalSet,
    LLMJudge,
    MaxLength,
    ReflectiveOptimizer,
    RunRecord,
    TextArtifact,
)
from recension.models import Message

STARTING_SKILL = """\
# Writing executive summaries

Write a summary of the supplied report. Try to keep it useful for a busy
reader and do not leave out anything that seems important.
"""

GOOD_REVISION = """\
# Writing executive summaries

Lead with the single most important finding in the first sentence.
Then give at most three supporting points, each with its number or date.
Cut everything a decision-maker would not act on. Maximum 80 words.
"""

RUBRIC = (
    "Score how well the summary serves a busy executive: leads with the main "
    "finding, concrete figures preserved, no filler, under 80 words."
)

VALIDATION_REPORT = (
    "Quarterly churn fell from 9.1% to 6.4% after the onboarding redesign, "
    "while support tickets per account dropped 22% and NPS rose four points."
)

RECORDS = [
    {"id": "t1", "split": "train",
     "input": "Revenue grew 14% year over year, driven by the EU expansion; margins "
              "compressed 2 points on logistics costs."},
    {"id": "t2", "split": "train",
     "input": "The migration finished four weeks late, but error budgets held and "
              "p99 latency improved 18% after the cache rewrite."},
    {"id": "v1", "split": "validation", "input": VALIDATION_REPORT},
    {"id": "v2", "split": "validation",
     "input": "Headcount stayed flat while shipped features doubled; attrition in "
              "the platform team remains the main delivery risk."},
    # A locked test split: a report never seen during optimization, judged once.
    {"id": "x1", "split": "test",
     "input": "Cloud spend rose 31% after the migration, but unit cost per request fell "
              "12% and reserved-instance coverage reached 80%."},
]


def mock_script(messages: list[Message]) -> str:
    """Offline stand-in for both the task model and the judge.

    The judge grades the *output*, which depends only on whether the artifact
    gave the model a concrete structure to follow, the same causal chain a
    real model-graded run exhibits.
    """
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    user = next(m["content"] for m in messages if m["role"] == "user")
    if user.startswith("You are grading a model output"):
        # Judge call (LLMJudge sends a single user message).
        return "9" if "MAIN FINDING:" in user else "4"
    if system.startswith("You analyze why"):
        return (
            "The instructions give no structure: nothing says what to lead with, "
            "how many points to keep, or any length limit, so summaries ramble."
        )
    if system.startswith("You revise text artifacts"):
        index = int(re.search(r"Propose revision (\d+)", user).group(1))  # type: ignore[union-attr]
        artifact = re.search(r"<artifact>\n(.*?)\n</artifact>", user, re.DOTALL).group(1)  # type: ignore[union-attr]
        variants = {
            1: "# Writing executive summaries\n\nSummarize the supplied report briefly "
               "and clearly, in plain language a busy reader can follow.",
            2: GOOD_REVISION,
            # Candidate 3 cheats: it pastes a validation report into the skill
            # file as a "worked example". The leakage check flags it.
            3: f"{artifact}\nFor example, summarize reports like: {VALIDATION_REPORT}",
        }
        return f"<revised_artifact>\n{variants[index]}\n</revised_artifact>"
    # Task call: structure in, structure out.
    if "Lead with the single most important finding" in system:
        return "MAIN FINDING: the key metric moved; three supporting points follow."
    return "This report discusses several topics which are summarized below at length."


def build_optimizer(
    real: bool = False, on_progress: Callable[[str], None] | None = None
) -> ReflectiveOptimizer:
    """Construct the optimizer for this example (shared by the CLI run and the web demo)."""
    model = get_model(real, mock_script)
    return ReflectiveOptimizer(
        artifact=TextArtifact.from_text(STARTING_SKILL, name="summary-skill"),
        evalset=EvalSet.from_records(RECORDS),
        objective=LLMJudge(model, rubric=RUBRIC),
        model=model,
        budget=Budget(candidates_per_round=3, rounds=2, diagnosis_depth=1, max_model_calls=300),
        seed=23,
        accept_significant=True,   # accept only a statistically significant validation gain
        guards=[MaxLength(70)],     # the structured rewrite must keep summaries tight
        on_progress=on_progress,
    )


def main(real: bool = False) -> RunRecord:
    return build_optimizer(real=real).run()


if __name__ == "__main__":
    args = parse_args(__doc__ or "")
    record = main(real=args.real)
    report(record, "skill_file")
