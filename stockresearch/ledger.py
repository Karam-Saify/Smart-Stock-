"""Utilities for combining method-level experiment outputs and run manifests."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

RESULT_COLUMNS = ["test_id", "ticker", "status", "method", "observations", "metrics_json", "note"]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".csv", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}-", suffix=".json", dir=path.parent)
    os.close(fd)
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_combined_ledger(
    frames: list[pd.DataFrame],
    output_dir: str | Path,
    experiment_metadata: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write the canonical result ledger, summary and optional provenance manifest."""
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
    if results.empty:
        summary = pd.DataFrame(columns=["test_id"])
    else:
        summary = results.groupby(["test_id", "status"]).size().unstack(fill_value=0).reset_index()

    _atomic_csv(results, dest / "experiment_results.csv")
    _atomic_csv(summary, dest / "experiment_summary.csv")
    if experiment_metadata is not None:
        manifest = dict(experiment_metadata)
        manifest["result_rows"] = int(len(results))
        manifest["status_counts"] = (
            results["status"].value_counts(dropna=False).sort_index().to_dict()
            if not results.empty else {}
        )
        _atomic_json(manifest, dest / "experiment_manifest.json")
    return results, summary
