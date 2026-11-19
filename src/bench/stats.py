"""Small statistics helpers: accuracy, means, a Wilson interval and a paired bootstrap.

Why a paired bootstrap for comparisons: the edited, baseline and control runs answer the same
questions. Taking the difference per question keeps that pairing, so the noise from "which
questions happen to be hard" cancels out and the interval is much tighter than comparing two
independent means. The bootstrap resamples questions (not runs) to estimate that interval.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Sequence

import numpy as np

# Fixed so that the same inputs always give the same interval. Change it only on purpose.
DEFAULT_SEED = 20260101


def accuracy(correct: Sequence[bool]) -> float:
    """Share of answers that are correct, as a fraction between 0 and 1."""
    if len(correct) == 0:
        raise ValueError("accuracy of zero answers is undefined")
    return float(np.mean(np.asarray(correct, dtype=float)))


def mean(values: Sequence[float]) -> float:
    if len(values) == 0:
        raise ValueError("mean of zero values is undefined")
    return float(np.mean(np.asarray(values, dtype=float)))


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Confidence interval for one proportion (e.g. accuracy of one run).

    Unlike the plain normal approximation, it stays inside 0 to 1 and behaves well near 0 or 100%.
    """
    if n == 0:
        raise ValueError("interval for zero answers is undefined")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half_width = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denominator
    return centre - half_width, centre + half_width


def bootstrap_mean_difference(
    a: Sequence[float],
    b: Sequence[float],
    confidence: float = 0.95,
    samples: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Paired bootstrap for mean(a) - mean(b), where a[i] and b[i] belong to the same question.

    Returns (difference, CI low, CI high). Questions are resampled with replacement, the same
    question index for both arrays each time, and the percentile interval of the resampled
    differences is returned.
    """
    first = np.asarray(a, dtype=float)
    second = np.asarray(b, dtype=float)
    if first.shape != second.shape:
        raise ValueError(f"paired arrays must have the same length, got {first.size} and {second.size}")
    if first.size == 0:
        raise ValueError("bootstrap of zero pairs is undefined")

    differences = first - second
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, differences.size, size=(samples, differences.size))
    resampled_means = differences[picks].mean(axis=1)

    tail = (1 - confidence) / 2
    low, high = np.quantile(resampled_means, [tail, 1 - tail])
    return float(differences.mean()), float(low), float(high)
