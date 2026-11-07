"""Utilities for combining method-level experiment outputs."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

RESULT_COLUMNS = ["test_id", "ticker", "status", "method", "observations", "metrics_json", "note"]


def write_combined_ledger(frames: list[pd.DataFrame], output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write one canonical ledger and summary from independent experiment engines."""
    normalized = []
    for frame in frames:
        if frame is None or frame.empty:
            continue
        missing = sorted(set(RESULT_COLUMNS) - set(frame.columns))
        if missing:
            raise ValueError(f"experiment frame is missing required columns: {missing}")
        normalized.append(frame.loc[:, RESULT_COLUMNS].copy())
    results = pd.concat(normalized, ignore_index=True) if normalized else pd.DataFrame(columns=RESULT_COLUMNS)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    results.to_csv(dest / "experiment_results.csv", index=False)
    if results.empty:
        summary = pd.DataFrame(columns=["test_id"])
    else:
        summary = results.groupby(["test_id", "status"]).size().unstack(fill_value=0).reset_index()
    summary.to_csv(dest / "experiment_summary.csv", index=False)
    return results, summary
