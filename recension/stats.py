"""Seeded bootstrap statistics for honest acceptance decisions.

The optimizer can require an accepted candidate's validation gain to be
*statistically significant*, not merely larger than an epsilon, so a candidate
that wins by noise is rejected. This module provides the paired-difference
bootstrap that backs that gate. It is pure stdlib (``random.Random``) and fully
deterministic given a seed, so a seeded run against ``MockModel`` stays
reproducible.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["BootstrapResult", "paired_bootstrap"]


@dataclass(frozen=True)
class BootstrapResult:
    """A paired-difference bootstrap of two aligned per-example score vectors.

    Attributes:
        mean_difference: Mean of ``candidate_i - incumbent_i`` over the
            validation examples (the observed per-example gain).
        ci_low: Lower bound of the ``1 - alpha`` confidence interval on the
            mean difference.
        ci_high: Upper bound of that interval.
        alpha: The significance level used (e.g. ``0.05`` for a 95% interval).
        n_resamples: Number of bootstrap resamples drawn.
    """

    mean_difference: float
    ci_low: float
    ci_high: float
    alpha: float
    n_resamples: int

    @property
    def significant(self) -> bool:
        """True when the interval excludes 0, i.e. the gain is significantly positive."""
        return self.ci_low > 0.0


def paired_bootstrap(
    incumbent: Sequence[float],
    candidate: Sequence[float],
    *,
    alpha: float = 0.05,
    n_resamples: int = 2000,
    seed: int | None = None,
) -> BootstrapResult:
    """Bootstrap a confidence interval on the mean paired score difference.

    Resamples the per-example differences ``candidate_i - incumbent_i`` with
    replacement to estimate a percentile confidence interval on their mean.

    Args:
        incumbent: Per-example validation scores of the incumbent.
        candidate: Per-example validation scores of the candidate, aligned with
            ``incumbent`` (same examples, same order).
        alpha: Significance level; the interval is ``1 - alpha``.
        n_resamples: Number of resamples.
        seed: Seed for the resampling RNG; pass one for reproducibility.

    Raises:
        ValueError: If the vectors differ in length or are empty.
    """
    if len(incumbent) != len(candidate):
        raise ValueError("paired bootstrap needs equal-length score vectors")
    n = len(incumbent)
    if n == 0:
        raise ValueError("cannot bootstrap an empty score vector")
    diffs = [c - i for i, c in zip(incumbent, candidate, strict=True)]
    mean_difference = sum(diffs) / n

    rng = random.Random(seed)
    resample_means: list[float] = []
    for _ in range(n_resamples):
        total = 0.0
        for _ in range(n):
            total += diffs[rng.randrange(n)]
        resample_means.append(total / n)
    resample_means.sort()

    lo_idx = int((alpha / 2) * n_resamples)
    hi_idx = int((1 - alpha / 2) * n_resamples) - 1
    hi_idx = max(lo_idx, min(hi_idx, n_resamples - 1))
    return BootstrapResult(
        mean_difference=mean_difference,
        ci_low=resample_means[lo_idx],
        ci_high=resample_means[hi_idx],
        alpha=alpha,
        n_resamples=n_resamples,
    )
