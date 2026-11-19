"""Tests for the statistics helpers (offline, deterministic)."""

import numpy as np
import pytest

from bench.stats import accuracy, bootstrap_mean_difference, mean, wilson_interval


def test_accuracy_and_mean():
    assert accuracy([True, False, True, True]) == pytest.approx(0.75)
    assert mean([1.0, 2.0, 6.0]) == pytest.approx(3.0)
    with pytest.raises(ValueError):
        accuracy([])


def test_wilson_interval_stays_inside_zero_and_one():
    low, high = wilson_interval(0, 10)
    assert low == pytest.approx(0.0, abs=1e-12)
    assert 0 < high < 0.35

    low, high = wilson_interval(50, 100)
    assert low < 0.5 < high
    assert (0.5 - low) == pytest.approx(high - 0.5)


def test_bootstrap_of_identical_arrays_is_centred_on_zero():
    values = [0.1, 0.4, 0.9, 0.2, 0.7, 0.5]
    diff, low, high = bootstrap_mean_difference(values, values)
    assert diff == 0.0
    assert low == 0.0 and high == 0.0
    assert low <= 0 <= high


def test_bootstrap_of_a_clear_shift_has_interval_above_zero():
    rng = np.random.default_rng(1)
    baseline = rng.uniform(0.2, 0.8, size=200)
    edited = baseline + 0.2 + rng.normal(0, 0.02, size=200)
    diff, low, high = bootstrap_mean_difference(edited, baseline)
    assert diff == pytest.approx(0.2, abs=0.01)
    assert low > 0
    assert high > low


def test_bootstrap_of_a_clear_drop_has_interval_below_zero():
    rng = np.random.default_rng(2)
    baseline = rng.uniform(0.2, 0.8, size=200)
    edited = baseline - 0.2 + rng.normal(0, 0.02, size=200)
    _, low, high = bootstrap_mean_difference(edited, baseline)
    assert high < 0


def test_bootstrap_is_reproducible_with_the_same_seed():
    a = [0.1, 0.9, 0.4, 0.6, 0.2]
    b = [0.3, 0.5, 0.2, 0.4, 0.1]
    assert bootstrap_mean_difference(a, b, seed=7) == bootstrap_mean_difference(a, b, seed=7)


def test_bootstrap_rejects_unpaired_lengths():
    with pytest.raises(ValueError, match="same length"):
        bootstrap_mean_difference([1.0, 2.0], [1.0])
