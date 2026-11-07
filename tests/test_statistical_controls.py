import numpy as np
import pandas as pd
import pytest

from stockresearch.statistical_controls import (
    PurgedWalkForwardSplit, adjust_pvalues, moving_block_bootstrap_mean_ci,
)


def test_multiple_testing_adjustments_are_bounded_and_monotone():
    p = np.array([0.04, 0.001, 0.03, 0.2, np.nan])
    for method in ("holm", "bonferroni", "fdr_bh", "fdr_by"):
        adjusted = adjust_pvalues(p, method)
        assert np.isnan(adjusted[-1])
        assert np.all((adjusted[:-1] >= p[:-1]) & (adjusted[:-1] <= 1))
    bh = adjust_pvalues(p, "fdr_bh")
    assert bh[1] <= bh[2] <= bh[0] <= bh[3]


def test_multiple_testing_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        adjust_pvalues([0.1, 1.2])
    with pytest.raises(ValueError):
        adjust_pvalues([0.1], "unknown")
    with pytest.raises(ValueError):
        adjust_pvalues([[0.1, 0.2]])


def test_walk_forward_is_chronological():
    idx = pd.bdate_range("2020-01-01", periods=120)
    X = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    cv = PurgedWalkForwardSplit(n_splits=4, test_size=15, min_train_size=30)
    for train, test in cv.split(X):
        assert train.max() < test.min()
        assert not set(train).intersection(test)
        assert len(test) == 15


def test_walk_forward_purges_overlapping_event_labels():
    idx = pd.bdate_range("2020-01-01", periods=100)
    X = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    label_end = pd.Series(idx, index=idx)
    label_end.iloc[60:80] = idx[80]
    cv = PurgedWalkForwardSplit(n_splits=1, test_size=20, min_train_size=20)
    train, test = next(cv.split(X, label_end=label_end))
    assert test[0] == 80
    assert (label_end.iloc[train].to_numpy() < idx[test[0]]).all()
    assert train.max() < test.min()


def test_walk_forward_rejects_misaligned_labels():
    idx = pd.bdate_range("2020-01-01", periods=80)
    X = pd.DataFrame({"x": np.arange(len(idx))}, index=idx)
    wrong = pd.Series(idx[:-1], index=idx[:-1])
    cv = PurgedWalkForwardSplit(n_splits=2)
    with pytest.raises(ValueError, match="exactly match"):
        list(cv.split(X, label_end=wrong))


def test_block_bootstrap_is_reproducible_and_well_formed():
    x = np.random.default_rng(8).normal(0.2, 1.0, size=300)
    a = moving_block_bootstrap_mean_ci(x, block_length=5, random_state=11)
    b = moving_block_bootstrap_mean_ci(x, block_length=5, random_state=11)
    assert a == b
    assert a["ci_lower"] <= a["mean"] <= a["ci_upper"]
    assert a["sample_size"] == 300


def test_block_bootstrap_rejects_invalid_settings():
    with pytest.raises(ValueError):
        moving_block_bootstrap_mean_ci([1, 2], block_length=1)
    with pytest.raises(ValueError):
        moving_block_bootstrap_mean_ci([1, 2, 3], block_length=0)
