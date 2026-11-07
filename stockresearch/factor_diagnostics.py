"""Cross-sectional factor diagnostics for research panels.

Inputs must be constructed upstream with point-in-time information and correctly
aligned forward returns. Outputs are descriptive diagnostics, not trade signals.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _validate_panels(factors: pd.DataFrame, forward_returns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not isinstance(factors, pd.DataFrame) or not isinstance(forward_returns, pd.DataFrame):
        raise TypeError("factors and forward_returns must be pandas DataFrames")
    if factors.empty or forward_returns.empty:
        raise ValueError("input panels must not be empty")
    if not factors.index.is_unique or not forward_returns.index.is_unique:
        raise ValueError("panel dates must be unique")
    if not factors.columns.is_unique or not forward_returns.columns.is_unique:
        raise ValueError("asset columns must be unique")
    common_dates = factors.index.intersection(forward_returns.index)
    common_assets = factors.columns.intersection(forward_returns.columns)
    if len(common_dates) == 0 or len(common_assets) < 3:
        raise ValueError("need common dates and at least three common assets")
    f = factors.loc[common_dates, common_assets].apply(pd.to_numeric, errors="coerce")
    r = forward_returns.loc[common_dates, common_assets].apply(pd.to_numeric, errors="coerce")
    f = f.replace([np.inf, -np.inf], np.nan)
    r = r.replace([np.inf, -np.inf], np.nan)
    return f.sort_index(), r.sort_index()


def information_coefficient(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    min_assets: int = 5,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Compute date-wise Spearman rank IC and a compact descriptive summary."""
    if min_assets < 3:
        raise ValueError("min_assets must be at least 3")
    f, r = _validate_panels(factors, forward_returns)
    rows = []
    for date in f.index:
        pair = pd.concat([f.loc[date].rename("factor"), r.loc[date].rename("forward_return")], axis=1).dropna()
        if len(pair) < min_assets or pair["factor"].nunique() < 2 or pair["forward_return"].nunique() < 2:
            rows.append({"date": date, "assets": len(pair), "rank_ic": np.nan, "status": "SKIPPED"})
            continue
        ic = stats.spearmanr(pair["factor"], pair["forward_return"]).statistic
        rows.append({"date": date, "assets": len(pair), "rank_ic": float(ic), "status": "OK"})
    result = pd.DataFrame(rows).set_index("date")
    valid = result.loc[result["status"] == "OK", "rank_ic"].dropna()
    summary = {
        "periods_total": int(len(result)),
        "periods_valid": int(len(valid)),
        "mean_rank_ic": float(valid.mean()) if len(valid) else float("nan"),
        "rank_ic_std": float(valid.std(ddof=1)) if len(valid) > 1 else float("nan"),
        "positive_ic_fraction": float((valid > 0).mean()) if len(valid) else float("nan"),
        "median_assets": int(result.loc[result["status"] == "OK", "assets"].median()) if len(valid) else 0,
    }
    return result, summary


def quantile_return_profile(
    factors: pd.DataFrame,
    forward_returns: pd.DataFrame,
    *,
    quantiles: int = 5,
    min_assets: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return date-by-quantile mean forward returns and an aggregate profile.

    Quantile assignments are cross-sectional within each date. Ties may produce
    fewer than the requested number of groups; dates with insufficient assets
    or fewer than two distinct factor values are marked SKIPPED.
    """
    if quantiles < 2:
        raise ValueError("quantiles must be at least 2")
    if min_assets < quantiles:
        raise ValueError("min_assets must be at least quantiles")
    f, r = _validate_panels(factors, forward_returns)
    rows = []
    for date in f.index:
        pair = pd.concat([f.loc[date].rename("factor"), r.loc[date].rename("forward_return")], axis=1).dropna()
        if len(pair) < min_assets or pair["factor"].nunique() < 2:
            rows.append({"date": date, "quantile": np.nan, "assets": len(pair), "mean_forward_return": np.nan, "status": "SKIPPED"})
            continue
        try:
            labels = pd.qcut(pair["factor"].rank(method="first"), q=quantiles, labels=False) + 1
        except ValueError:
            rows.append({"date": date, "quantile": np.nan, "assets": len(pair), "mean_forward_return": np.nan, "status": "SKIPPED"})
            continue
        for q, group in pair.assign(quantile=labels).groupby("quantile"):
            rows.append({"date": date, "quantile": int(q), "assets": int(len(group)), "mean_forward_return": float(group["forward_return"].mean()), "status": "OK"})
    detail = pd.DataFrame(rows)
    valid = detail.loc[detail["status"] == "OK"]
    profile = (valid.groupby("quantile")["mean_forward_return"]
               .agg(["mean", "std", "count"]).reset_index()
               .rename(columns={"mean": "average_period_return", "std": "period_return_std", "count": "periods"}))
    return detail, profile
