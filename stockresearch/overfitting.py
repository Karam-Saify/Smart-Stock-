"""Combinatorially symmetric cross-validation diagnostics.

This module evaluates the rank stability of a supplied matrix of strategy returns.
It does not create strategies, optimize parameters, or recommend allocations.
"""
from __future__ import annotations

from itertools import combinations
import math

import numpy as np
import pandas as pd


def probability_of_backtest_overfitting(
    returns: pd.DataFrame,
    *,
    n_slices: int = 8,
    min_observations_per_slice: int = 10,
    selection_metric: str = "mean",
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Estimate PBO using CSCV on an already-defined candidate return matrix.

    Rows are chronological observations and columns are candidate strategies.
    The sample is split into equal contiguous slices (remainder is discarded).
    For each choice of half the slices as in-sample, the best candidate under
    selection_metric is selected and ranked under the same metric out of sample.
    Supported metrics are arithmetic mean return and daily Sharpe ratio. The
    metric must match the research protocol's candidate-selection objective.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.empty or returns.shape[1] < 2:
        raise ValueError("returns must contain observations and at least two candidates")
    if not returns.index.is_unique or not returns.columns.is_unique:
        raise ValueError("return index and candidate columns must be unique")
    if n_slices < 4 or n_slices % 2:
        raise ValueError("n_slices must be an even integer of at least 4")
    if min_observations_per_slice < 2:
        raise ValueError("min_observations_per_slice must be at least 2")
    if selection_metric not in {"mean", "sharpe"}:
        raise ValueError("selection_metric must be 'mean' or 'sharpe'")
    numeric = returns.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if numeric.isna().any().any():
        raise ValueError("returns must be complete and finite; handle missingness explicitly")
    slice_size = len(numeric) // n_slices
    if slice_size < min_observations_per_slice:
        raise ValueError("not enough observations for the requested number of slices")
    usable = numeric.iloc[: slice_size * n_slices]
    slices = [usable.iloc[i * slice_size : (i + 1) * slice_size] for i in range(n_slices)]
    half = n_slices // 2
    rows = []
    for in_sample_ids in combinations(range(n_slices), half):
        in_set = set(in_sample_ids)
        out_sample_ids = tuple(i for i in range(n_slices) if i not in in_set)
        train = pd.concat([slices[i] for i in in_sample_ids], axis=0)
        test = pd.concat([slices[i] for i in out_sample_ids], axis=0)
        def score(frame: pd.DataFrame) -> pd.Series:
            mean = frame.mean(axis=0)
            if selection_metric == "mean":
                return mean
            std = frame.std(axis=0, ddof=1).replace(0.0, np.nan)
            return (mean / std).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)

        train_score = score(train)
        selected_label = train_score.idxmax()
        selected = str(selected_label)
        test_score = score(test)
        # Keep the original column label for indexing (asset names need not be strings).
        # Percentile 1 is best; percentile 0 is worst.
        percentile = float(test_score.rank(method="average", pct=True).loc[selected_label])
        # PBO event: selected candidate falls below the OOS median.
        rows.append({
            "in_sample_slices": ",".join(map(str, in_sample_ids)),
            "selected_candidate": selected,
            "in_sample_mean": float(train_score.loc[selected_label]),
            "out_of_sample_mean": float(test_score.loc[selected_label]),
            "oos_percentile": percentile,
            "below_oos_median": bool(percentile < 0.5),
        })
    detail = pd.DataFrame(rows)
    pbo = float(detail["below_oos_median"].mean())
    summary = {
        "pbo": pbo,
        "splits": int(len(detail)),
        "candidates": int(numeric.shape[1]),
        "selection_metric": selection_metric,
        "observations_used": int(len(usable)),
        "observations_discarded_remainder": int(len(numeric) - len(usable)),
        "interpretation": "descriptive_selection_instability_diagnostic",
    }
    return detail, summary
