import numpy as np
import pandas as pd
import pytest

from stockresearch.overfitting import probability_of_backtest_overfitting


def test_pbo_is_reproducible_and_enumerates_combinations():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame(rng.normal(0.001, 0.02, size=(160, 5)))
    detail, summary = probability_of_backtest_overfitting(frame, n_slices=8)
    assert len(detail) == 70
    assert summary["splits"] == 70
    assert 0.0 <= summary["pbo"] <= 1.0
    assert detail["oos_percentile"].between(0, 1).all()


def test_pbo_detects_obvious_stable_ranking():
    base = np.tile(np.linspace(-0.01, 0.02, 20), 8)
    frame = pd.DataFrame({"best": base + 0.01, "middle": base, "worst": base - 0.01})
    detail, summary = probability_of_backtest_overfitting(frame, n_slices=8)
    assert summary["pbo"] == 0.0
    assert (detail["selected_candidate"] == "best").all()


@pytest.mark.parametrize("kwargs", [
    {"n_slices": 3},
    {"n_slices": 5},
    {"min_observations_per_slice": 1},
])
def test_invalid_settings(kwargs):
    frame = pd.DataFrame(np.ones((80, 3)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(frame, **kwargs)


def test_rejects_missing_values_and_too_few_candidates():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(pd.DataFrame({"a": [1.0] * 80, "b": [np.nan] * 80}))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(pd.DataFrame({"a": [1.0] * 80}))
