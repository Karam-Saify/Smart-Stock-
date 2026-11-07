import numpy as np
import pandas as pd
import pytest

from stockresearch.factor_diagnostics import information_coefficient, quantile_return_profile


def panels():
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    assets = list("ABCDEFGH")
    factors = pd.DataFrame([np.arange(8), np.arange(8), np.arange(8), np.arange(8)], index=dates, columns=assets)
    returns = factors.copy()
    returns.iloc[1] = -returns.iloc[1]
    returns.iloc[2] = np.nan
    return factors, returns


def test_information_coefficient_reports_datewise_rank_correlations():
    f, r = panels()
    detail, summary = information_coefficient(f, r, min_assets=5)
    assert detail.loc[f.index[0], "rank_ic"] == pytest.approx(1.0)
    assert detail.loc[f.index[1], "rank_ic"] == pytest.approx(-1.0)
    assert detail.loc[f.index[2], "status"] == "SKIPPED"
    assert summary["periods_valid"] == 3


def test_quantile_profile_has_requested_groups():
    f, r = panels()
    detail, profile = quantile_return_profile(f, r, quantiles=4, min_assets=8)
    assert set(profile["quantile"]) == {1, 2, 3, 4}
    assert len(detail.loc[detail["date"] == f.index[0]]) == 4


def test_rejects_misaligned_or_invalid_settings():
    f, r = panels()
    with pytest.raises(ValueError):
        information_coefficient(f, r, min_assets=2)
    with pytest.raises(ValueError):
        quantile_return_profile(f, r, quantiles=5, min_assets=4)


def test_requires_shared_assets_and_dates():
    f = pd.DataFrame({"A": [1.0], "B": [2.0], "C": [3.0]}, index=[pd.Timestamp("2024-01-01")])
    r = pd.DataFrame({"X": [1.0], "Y": [2.0], "Z": [3.0]}, index=f.index)
    with pytest.raises(ValueError):
        information_coefficient(f, r)
