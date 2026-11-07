"""Pre-research data integrity diagnostics for adjusted-price panels and raw bars."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _raw_bar_checks(path: Path) -> dict:
    """Inspect raw rows before date normalization can conceal duplicate timestamps."""
    try:
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception as exc:
        return {"raw_file_status": "unreadable", "raw_file_error": str(exc)}

    checks = {
        "raw_file_status": "read",
        "raw_rows": int(len(raw)),
        "raw_duplicate_dates": int(raw.index.duplicated(keep=False).sum()),
        "raw_invalid_dates": int(pd.isna(pd.to_datetime(raw.index, errors="coerce")).sum()),
    }
    numeric = {}
    for col in ("Open", "High", "Low", "Close", "Adj Close", "Volume"):
        if col in raw:
            numeric[col] = pd.to_numeric(raw[col], errors="coerce")

    for col in ("Open", "High", "Low", "Close", "Adj Close"):
        if col in numeric:
            values = numeric[col]
            checks[f"nonpositive_{col.lower().replace(' ', '_')}"] = int((values <= 0).sum())
            checks[f"missing_{col.lower().replace(' ', '_')}"] = int(values.isna().sum())

    if all(col in numeric for col in ("Open", "High", "Low", "Close")):
        o, h, l, c = (numeric[col] for col in ("Open", "High", "Low", "Close"))
        valid = o.notna() & h.notna() & l.notna() & c.notna()
        checks["ohlc_inconsistent_rows"] = int(
            (valid & ((h < pd.concat([o, l, c], axis=1).max(axis=1)) |
                      (l > pd.concat([o, h, c], axis=1).min(axis=1)))).sum()
        )
    if "Volume" in numeric:
        volume = numeric["Volume"]
        checks["negative_volume_rows"] = int((volume < 0).sum())
        checks["zero_volume_rows"] = int((volume == 0).sum())

    if "Adj Close" in numeric and "Close" in numeric:
        close = numeric["Close"].where(numeric["Close"] > 0)
        ratio = numeric["Adj Close"] / close
        checks["adjustment_ratio_min"] = float(ratio.min()) if ratio.notna().any() else np.nan
        checks["adjustment_ratio_max"] = float(ratio.max()) if ratio.notna().any() else np.nan
        checks["adjustment_ratio_gt_2_rows"] = int((ratio > 2).sum())
        checks["adjustment_ratio_lt_0_5_rows"] = int((ratio < 0.5).sum())

    dates = pd.DatetimeIndex(pd.to_datetime(raw.index, errors="coerce")).dropna().sort_values()
    if len(dates) > 1:
        gaps = dates.to_series().diff().dt.days
        # Calendar-gap screen only. It deliberately does not call ordinary weekends/holidays errors.
        checks["calendar_gaps_over_7_days"] = int((gaps > 7).sum())
        checks["max_calendar_gap_days"] = int(gaps.max())
    else:
        checks["calendar_gaps_over_7_days"] = 0
        checks["max_calendar_gap_days"] = 0
    return checks


def audit_panels(data_dir):
    """Write a per-instrument quality report; diagnostics never silently repair data."""
    root = Path(data_dir)
    panel_path = root / "panel_adjclose.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"{panel_path} missing; run download first")

    px = pd.read_csv(panel_path, index_col=0, parse_dates=True)
    metadata = _read_json(root / "metadata.json")
    download = _read_json(root / "download_manifest.json")
    instrument_meta = metadata.get("instruments", {})
    download_ok = download.get("downloaded", {})
    download_failures = download.get("failures", {})
    raw_dir = root / "raw"
    out = []

    for ticker in px:
        s = pd.to_numeric(px[ticker], errors="coerce")
        positive = s.where(s > 0)
        returns = np.log(positive).diff()
        valid = s.notna()
        raw_path = raw_dir / f"{ticker.replace('^', '_').replace('=', '_').replace('/', '_')}.csv"
        raw_checks = _raw_bar_checks(raw_path) if raw_path.exists() else {
            "raw_file_status": "missing",
            "raw_file_error": None,
        }
        info = instrument_meta.get(ticker, {})
        out.append({
            "ticker": ticker,
            "observations": int(valid.sum()),
            "missing_adjusted_prices": int(s.isna().sum()),
            "nonpositive_adjusted_prices": int((s <= 0).sum()),
            "duplicate_panel_dates": int(px.index.duplicated(keep=False).sum()),
            "return_outliers_abs_gt_40pct": int((returns.abs() > .4).sum()),
            "max_abs_daily_log_return": float(returns.abs().max()) if returns.notna().any() else np.nan,
            "first_date": str(s.first_valid_index().date()) if valid.any() else None,
            "last_date": str(s.last_valid_index().date()) if valid.any() else None,
            "download_status": "failed" if ticker in download_failures else (
                "downloaded" if ticker in download_ok else info.get("status", "unknown")
            ),
            "source_first_date": info.get("first"),
            "source_last_date": info.get("last"),
            **raw_checks,
        })

    report = pd.DataFrame(out)
    report.to_csv(root / "quality_audit.csv", index=False)
    return report
