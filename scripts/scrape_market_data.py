#!/usr/bin/env python3
"""Download real daily OHLCV and corporate-action history for the research universe.

This is an operational entry point over the project's existing Yahoo Finance
adapter. Raw vendor rows, download metadata, adjusted-price panels and a
quality audit are retained under the selected data directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from datetime import date
from pathlib import Path
from typing import Sequence

from stockresearch import data, universe
from stockresearch.quality import audit_panels


TICKER_PATTERN = re.compile(r"^[A-Z0-9.^=_-]+$")


def parse_tickers(value: str | None) -> list[str]:
    """Parse a comma-separated symbol list; preserve provider-native symbols."""
    if value is None or not value.strip():
        return universe.all_tickers()
    tickers = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not tickers:
        raise argparse.ArgumentTypeError("Provide at least one non-empty ticker")
    invalid = [ticker for ticker in tickers if not TICKER_PATTERN.fullmatch(ticker)]
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid ticker symbol(s): {', '.join(invalid)}")
    if len(set(tickers)) != len(tickers):
        raise argparse.ArgumentTypeError("Ticker list contains duplicates")
    return tickers


def validate_dates(start: str | None, end: str | None) -> None:
    """Validate ISO dates and Yahoo's exclusive end-date convention."""
    try:
        start_date = date.fromisoformat(start) if start else None
        end_date = date.fromisoformat(end) if end else None
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Dates must use YYYY-MM-DD format") from exc
    if start_date and end_date and end_date <= start_date:
        raise argparse.ArgumentTypeError("End date must be later than start date")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch real daily market data and build auditable research panels."
    )
    parser.add_argument("--data-dir", default="data", help="Output directory (default: data)")
    parser.add_argument("--start", help="First date, YYYY-MM-DD (default: project start date)")
    parser.add_argument(
        "--end",
        help="Exclusive end date, YYYY-MM-DD (default: latest available vendor data)",
    )
    parser.add_argument(
        "--tickers",
        help="Comma-separated symbols; default is the full 48-instrument research universe",
    )
    parser.add_argument(
        "--no-macro",
        action="store_true",
        help="Skip macro series and FRED CPI retrieval",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=60,
        help="Minimum raw daily rows required per requested symbol (default: 60)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Return success when at least one symbol is usable, even if others fail",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        tickers = parse_tickers(args.tickers)
        validate_dates(args.start, args.end)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))
    if args.min_observations < 2:
        parser.error("--min-observations must be at least 2")

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    root = Path(args.data_dir)
    root.mkdir(parents=True, exist_ok=True)

    failures = data.download_all(
        data_dir=root,
        start=args.start,
        end=args.end,
        tickers=tickers,
        include_macro=not args.no_macro,
    )
    audit = audit_panels(root)

    manifest_path = root / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloaded = manifest.get("downloaded", {})
    failed_map = manifest.get("failures", {})
    usable = {
        ticker: item for ticker, item in downloaded.items()
        if int(item.get("rows", 0)) >= args.min_observations
    }
    too_short = {
        ticker: int(item.get("rows", 0))
        for ticker, item in downloaded.items()
        if int(item.get("rows", 0)) < args.min_observations
    }
    missing_or_failed = sorted(set(tickers) - set(usable))
    summary = {
        "provider": "Yahoo Finance via yfinance",
        "data_type": "daily OHLCV, adjusted close, dividends and splits where supplied",
        "requested_tickers": tickers,
        "requested_count": len(tickers),
        "downloaded_count": len(downloaded),
        "usable_count": len(usable),
        "failed_tickers": sorted(set(failures) | set(failed_map)),
        "insufficient_history": too_short,
        "missing_or_unusable": missing_or_failed,
        "minimum_observations": args.min_observations,
        "macro_requested": not args.no_macro,
        "data_directory": str(root.resolve()),
        "artifacts": [
            "raw/*.csv",
            "panel_adjclose.csv",
            "panel_close.csv",
            "panel_volume.csv",
            "metadata.json",
            "download_manifest.json",
            "quality_audit.csv",
        ],
        "quality_rows": int(len(audit)),
        "point_in_time_status": "not established; vendor history may be revised",
        "survivorship_warning": "The configured universe is a present-day sample, not a survivorship-free historical universe.",
    }
    summary_path = root / "scrape_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if not usable:
        return 2
    if missing_or_failed and not args.allow_partial:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
