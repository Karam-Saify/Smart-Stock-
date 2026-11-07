import numpy as np
import pandas as pd

from stockresearch.uncertainty import build_return_uncertainty


def _panel(path, *, missing=False):
    index = pd.bdate_range("2020-01-01", periods=180)
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0005, 0.01, len(index) - 1)
    prices = 100 * np.exp(np.r_[0.0, np.cumsum(returns)])
    frame = pd.DataFrame({"AAA": prices, "BBB": prices * 1.5}, index=index)
    if missing:
        frame.loc[index[50], "BBB"] = np.nan
    frame.to_csv(path / "panel_adjclose.csv")
    return frame


def test_moving_block_uncertainty_reports_mean_daily_log_return(tmp_path):
    panel = _panel(tmp_path)
    result = build_return_uncertainty(
        tmp_path, ["AAA"], block_length=5, n_bootstrap=200,
        confidence=0.95, random_state=12,
    )
    row = result.iloc[0]
    expected = np.diff(np.log(panel["AAA"].to_numpy())).mean()

    assert row["status"] == "OK"
    assert row["estimand"] == "mean daily log return"
    assert row["observations"] == 179
    assert row["mean_daily_log_return"] == expected
    assert row["ci_lower"] <= row["ci_upper"]
    assert row["annualization"] == "none; daily estimand only"


def test_incomplete_price_history_is_skipped_not_compressed(tmp_path):
    _panel(tmp_path, missing=True)
    result = build_return_uncertainty(tmp_path, ["BBB"])
    assert result.loc[0, "status"] == "SKIPPED"
    assert "do not compress gaps" in result.loc[0, "note"] or "no rows were silently removed" in result.loc[0, "note"]


def test_unknown_ticker_is_skipped(tmp_path):
    _panel(tmp_path)
    result = build_return_uncertainty(tmp_path, ["MISSING"])
    assert result.loc[0, "status"] == "SKIPPED"
    assert "absent" in result.loc[0, "note"]
