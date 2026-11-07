"""Statistical safeguards for repeated and time-dependent research.

These utilities are model-agnostic. They do not create trading signals; they
reduce common inference errors in exploratory research.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd


def adjust_pvalues(pvalues: Sequence[float], method: str = "holm") -> np.ndarray:
    """Adjust a family of p-values using Holm, Bonferroni, BH, or BY."""
    valid_methods = {"holm", "bonferroni", "fdr_bh", "fdr_by"}
    if method not in valid_methods:
        raise ValueError(f"method must be one of {sorted(valid_methods)}")
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvalues must be one-dimensional")
    finite = np.isfinite(p)
    if np.any((p[finite] < 0) | (p[finite] > 1)):
        raise ValueError("finite p-values must lie in [0, 1]")
    result = np.full(p.shape, np.nan, dtype=float)
    vals = p[finite]
    m = len(vals)
    if not m:
        return result
    order = np.argsort(vals, kind="mergesort")
    ranked = vals[order]
    ranks = np.arange(1, m + 1, dtype=float)
    if method == "bonferroni":
        adjusted = np.minimum(1.0, ranked * m)
    elif method == "holm":
        adjusted = np.minimum(1.0, np.maximum.accumulate((m - ranks + 1) * ranked))
    else:
        dependence_factor = 1.0 if method == "fdr_bh" else float(np.sum(1.0 / ranks))
        candidates = ranked * m * dependence_factor / ranks
        adjusted = np.minimum(1.0, np.minimum.accumulate(candidates[::-1])[::-1])
    restored = np.empty(m, dtype=float)
    restored[order] = adjusted
    result[finite] = restored
    return result


def _validated_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    dt = pd.DatetimeIndex(pd.to_datetime(index, errors="raise"))
    if dt.hasnans or not dt.is_monotonic_increasing or dt.has_duplicates:
        raise ValueError("index must contain valid, strictly increasing unique timestamps")
    return dt


class PurgedWalkForwardSplit:
    """Expanding-window chronological CV with event-label overlap purging.

    Each fold trains only on observations before its test block. If label_end is
    provided, training rows whose outcome interval reaches the first test
    timestamp are removed. label_end must be indexed exactly like X.
    """

    def __init__(self, n_splits: int = 5, test_size: int | None = None,
                 min_train_size: int | None = None, gap: int = 0):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if test_size is not None and test_size < 1:
            raise ValueError("test_size must be >= 1")
        if min_train_size is not None and min_train_size < 1:
            raise ValueError("min_train_size must be >= 1")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        self.n_splits = int(n_splits)
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = int(gap)

    def split(self, X: pd.DataFrame | pd.Series,
              label_end: pd.Series | None = None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        if not hasattr(X, "index"):
            raise TypeError("X must be a pandas object with a DatetimeIndex")
        index = _validated_datetime_index(X.index)
        n = len(index)
        size = n // (self.n_splits + 1) if self.test_size is None else int(self.test_size)
        if size < 1 or self.n_splits * size >= n:
            raise ValueError("test blocks must leave at least one training observation")
        first_test = n - self.n_splits * size
        min_train = self.min_train_size or max(2, first_test // 2)
        if first_test < min_train:
            raise ValueError(f"initial training window ({first_test}) is smaller than min_train_size ({min_train})")
        ends = None
        if label_end is not None:
            if not isinstance(label_end, pd.Series):
                raise TypeError("label_end must be a pandas Series")
            if not label_end.index.equals(X.index):
                raise ValueError("label_end index must exactly match X.index")
            parsed = pd.to_datetime(label_end, errors="coerce")
            if parsed.isna().any():
                raise ValueError("label_end cannot contain missing or invalid timestamps")
            if (parsed.to_numpy() < index.to_numpy()).any():
                raise ValueError("a label end cannot precede its observation timestamp")
            ends = parsed.to_numpy()
        for fold in range(self.n_splits):
            test_start = first_test + fold * size
            test_stop = test_start + size
            train = np.arange(max(0, test_start - self.gap), dtype=int)
            test = np.arange(test_start, test_stop, dtype=int)
            if ends is not None and len(train):
                train = train[ends[train] < index[test_start].to_datetime64()]
            if len(train) < min_train:
                raise ValueError(f"fold {fold} retains {len(train)} training rows after purging; minimum is {min_train}")
            yield train, test


def moving_block_bootstrap_mean_ci(values: Sequence[float], *, block_length: int,
                                   n_bootstrap: int = 2000, confidence: float = 0.95,
                                   random_state: int = 17) -> dict[str, float | int]:
    """Percentile CI for a mean using non-circular contiguous moving blocks."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    x = x[np.isfinite(x)]
    if len(x) < 3:
        raise ValueError("values must contain at least three finite observations")
    if not 1 <= block_length <= len(x):
        raise ValueError("block_length must be between 1 and sample length")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    rng = np.random.default_rng(random_state)
    n = len(x)
    n_blocks = int(np.ceil(n / block_length))
    max_start = n - block_length
    estimates = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([x[s:s + block_length] for s in starts])[:n]
        estimates[b] = np.mean(sample)
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(estimates, [alpha, 1.0 - alpha])
    return {"sample_size": int(n), "block_length": int(block_length),
            "n_bootstrap": int(n_bootstrap), "confidence": float(confidence),
            "mean": float(np.mean(x)), "ci_lower": float(lower), "ci_upper": float(upper)}
