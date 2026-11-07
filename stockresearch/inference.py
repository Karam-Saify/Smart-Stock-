"""Create explicit multiple-testing adjusted inference tables from engine outputs."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .statistical_controls import adjust_pvalues

P_VALUE_KEYS = {"pvalue", "p_value", "p-val", "p_val"}


def _walk_pvalues(value: Any, prefix: str = ""):
    if not isinstance(value, dict):
        return
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if str(key).lower() in P_VALUE_KEYS or str(key).lower().endswith(("_pvalue", "_p_value")):
            try:
                number = float(item)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number) and 0.0 <= number <= 1.0:
                yield path, number
        elif isinstance(item, dict):
            yield from _walk_pvalues(item, path)


def build_inference_table(results: pd.DataFrame, adjustment: str = "holm") -> pd.DataFrame:
    """Adjust each declared operational family across tickers for a fixed engine metric.

    A family is (test_id, method, nested metric path). This avoids mixing distinct
    null hypotheses, while controlling repeated ticker-level tests for the same
    method output. Cross-method and cross-question multiplicity remains the
    researcher's responsibility and is stated explicitly in each row.
    """
    if adjustment not in {"holm", "bonferroni", "fdr_bh", "fdr_by", "none"}:
        raise ValueError("adjustment must be holm, bonferroni, fdr_bh, fdr_by, or none")

    rows = []
    if results is None or results.empty:
        return pd.DataFrame(columns=[
            "test_id", "method", "metric_path", "ticker", "raw_pvalue",
            "adjusted_pvalue", "adjustment", "family_id", "family_size",
        ])

    required = {"test_id", "ticker", "status", "method", "metrics_json"}
    missing = sorted(required - set(results.columns))
    if missing:
        raise ValueError(f"results frame is missing required columns: {missing}")

    for record in results.to_dict("records"):
        if str(record.get("status", "")).upper() != "OK":
            continue
        raw = record.get("metrics_json")
        try:
            metrics = json.loads(raw) if isinstance(raw, str) else raw
        except (TypeError, ValueError):
            continue
        if not isinstance(metrics, dict):
            continue
        for metric_path, pvalue in _walk_pvalues(metrics):
            family_id = "|".join([
                str(record["test_id"]), str(record["method"]), metric_path,
            ])
            rows.append({
                "test_id": str(record["test_id"]),
                "method": str(record["method"]),
                "metric_path": metric_path,
                "ticker": str(record["ticker"]),
                "raw_pvalue": pvalue,
                "family_id": family_id,
            })

    columns = [
        "test_id", "method", "metric_path", "ticker", "raw_pvalue",
        "adjusted_pvalue", "adjustment", "family_id", "family_size",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    frame = pd.DataFrame(rows)
    frame["adjusted_pvalue"] = np.nan
    frame["adjustment"] = adjustment
    frame["family_size"] = 0
    for _, indices in frame.groupby("family_id", sort=True).groups.items():
        positions = list(indices)
        pvalues = frame.loc[positions, "raw_pvalue"].to_numpy(dtype=float)
        frame.loc[positions, "family_size"] = len(positions)
        if adjustment != "none":
            frame.loc[positions, "adjusted_pvalue"] = adjust_pvalues(
                pvalues, method=adjustment
            )
    return frame.loc[:, columns].sort_values(
        ["test_id", "method", "metric_path", "ticker"]
    ).reset_index(drop=True)


def write_inference_table(results: pd.DataFrame, output_dir: str | Path,
                          adjustment: str = "holm") -> pd.DataFrame:
    """Persist adjusted p-values as a separate, auditable output."""
    frame = build_inference_table(results, adjustment=adjustment)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest / "experiment_inference.csv", index=False)
    return frame
