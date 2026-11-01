"""Independent integrity gates for downloaded or synthetic adjusted-price panels.

This module diagnoses data quality; it never silently repairs, imputes, or drops data.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PanelIssue:
    severity: str
    ticker: str
    code: str
    detail: str


def audit_price_panel(
    prices: pd.DataFrame,
    *,
    max_missing_fraction: float = 0.05,
    extreme_log_return: float = 0.40,
) -> pd.DataFrame:
    """Return one row per detected issue; an empty frame means no configured issue.

    Dates are checked before normalization so duplicate and unsorted input cannot
    be concealed. Missing observations are measured against the panel's calendar
    index (not an exchange calendar), so interpret them as screening flags.
    """
    if not isinstance(prices, pd.DataFrame):
        raise TypeError("prices must be a pandas DataFrame")
    if prices.empty or len(prices.columns) == 0:
        raise ValueError("price panel must contain at least one row and one ticker")
    if not 0 <= max_missing_fraction <= 1:
        raise ValueError("max_missing_fraction must be between 0 and 1")
    if extreme_log_return <= 0:
        raise ValueError("extreme_log_return must be positive")

    issues: list[PanelIssue] = []
    idx = pd.to_datetime(prices.index, errors="coerce")
    if idx.isna().any():
        issues.append(PanelIssue("error", "*", "invalid_dates", "Index contains unparseable dates"))
    if idx.has_duplicates:
        issues.append(PanelIssue("error", "*", "duplicate_dates", f"{int(idx.duplicated().sum())} duplicate date rows"))
    if not idx.is_monotonic_increasing:
        issues.append(PanelIssue("error", "*", "unsorted_dates", "Dates are not strictly chronological"))

    for ticker in prices.columns:
        raw = prices[ticker]
        series = pd.to_numeric(raw, errors="coerce")
        non_numeric = int(raw.notna().sum() - series.notna().sum())
        if non_numeric:
            issues.append(PanelIssue("error", str(ticker), "non_numeric_prices", f"{non_numeric} non-numeric values"))
        nonpositive = int((series.notna() & (series <= 0)).sum())
        if nonpositive:
            issues.append(PanelIssue("error", str(ticker), "nonpositive_prices", f"{nonpositive} non-positive prices"))
        missing_fraction = float(series.isna().mean())
        if missing_fraction > max_missing_fraction:
            issues.append(PanelIssue("warning", str(ticker), "high_missingness", f"{missing_fraction:.1%} missing; threshold {max_missing_fraction:.1%}"))
        valid = series.where(series > 0)
        returns = np.log(valid).diff()
        extreme = int((returns.abs() > extreme_log_return).sum())
        if extreme:
            issues.append(PanelIssue("warning", str(ticker), "extreme_returns", f"{extreme} absolute log returns exceed {extreme_log_return:.2f}"))
        if valid.notna().sum() < 2:
            issues.append(PanelIssue("error", str(ticker), "insufficient_history", "Fewer than two valid positive prices"))

    return pd.DataFrame([asdict(issue) for issue in issues], columns=["severity", "ticker", "code", "detail"])


def assert_research_ready(prices: pd.DataFrame) -> dict[str, Any]:
    """Raise on structural errors; return summary while preserving warnings."""
    report = audit_price_panel(prices)
    errors = report.loc[report["severity"] == "error"]
    if not errors.empty:
        summary = "; ".join(f"{r.ticker}:{r.code}" for r in errors.itertuples())
        raise ValueError(f"Price panel failed integrity gates: {summary}")
    return {
        "rows": int(len(prices)),
        "tickers": int(len(prices.columns)),
        "start": str(pd.to_datetime(prices.index).min().date()),
        "end": str(pd.to_datetime(prices.index).max().date()),
        "warnings": report.loc[report["severity"] == "warning"].to_dict("records"),
    }
