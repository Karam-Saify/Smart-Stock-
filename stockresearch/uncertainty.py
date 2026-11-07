"""Dependence-aware uncertainty summaries for adjusted-price research panels."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .statistical_controls import moving_block_bootstrap_mean_ci


def build_return_uncertainty(
    data_dir: str | Path,
    tickers: list[str],
    *,
    block_length: int = 5,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Estimate a moving-block-bootstrap CI for each ticker's mean daily log return.

    Requires a complete, positive adjusted-price series for the selected sample.
    Missing interior prices are not dropped because doing so would join non-adjacent
    returns into artificial contiguous blocks. This is a descriptive historical
    estimand, not a forecast or an investment-performance guarantee.
    """
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    panel_path = Path(data_dir) / "panel_adjclose.csv"
    if not panel_path.is_file():
        raise FileNotFoundError(f"Adjusted-price panel not found: {panel_path}")
    panel = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("Adjusted-price panel must have unique, increasing dates")

    rows = []
    for ticker in tickers:
        base = {
            "ticker": ticker,
            "estimand": "mean daily log return",
            "confidence": confidence,
            "block_length": block_length,
            "n_bootstrap": n_bootstrap,
            "method": "non-circular moving-block percentile bootstrap",
            "status": "OK",
            "note": "",
        }
        if ticker not in panel.columns:
            rows.append({**base, "status": "SKIPPED", "note": "Ticker is absent from the adjusted-price panel"})
            continue

        prices = pd.to_numeric(panel[ticker], errors="coerce")
        observed = prices.dropna()
        if observed.empty:
            rows.append({**base, "status": "SKIPPED", "note": "No valid adjusted-price observations"})
            continue
        # Leading/trailing NaNs commonly represent pre-listing/post-coverage dates.
        # Trim only those boundaries; interior gaps would break block adjacency.
        prices = prices.loc[observed.index[0]:observed.index[-1]]
        if prices.isna().any() or (prices <= 0).any():
            rows.append({
                **base, "status": "SKIPPED",
                "note": "Interior missing or nonpositive adjusted prices; no observations were compressed",
            })
            continue
        returns = np.diff(np.log(prices.to_numpy(dtype=float)))
        if len(returns) < max(30, block_length * 4):
            rows.append({**base, "status": "SKIPPED", "note": "Insufficient contiguous return observations"})
            continue
        try:
            ci = moving_block_bootstrap_mean_ci(
                returns, block_length=block_length, n_bootstrap=n_bootstrap,
                confidence=confidence, random_state=random_state,
            )
            rows.append({
                **base,
                "observations": ci["sample_size"],
                "mean_daily_log_return": ci["mean"],
                "ci_lower": ci["ci_lower"],
                "ci_upper": ci["ci_upper"],
                "annualization": "none; daily estimand only",
            })
        except ValueError as exc:
            rows.append({**base, "status": "SKIPPED", "note": str(exc)})

    return pd.DataFrame(rows)


def write_return_uncertainty(data_dir: str | Path, output_dir: str | Path,
                             tickers: list[str], **kwargs) -> pd.DataFrame:
    """Write uncertainty results separately from engine p-values and point metrics."""
    frame = build_return_uncertainty(data_dir, tickers, **kwargs)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest / "experiment_uncertainty.csv", index=False)
    return frame
