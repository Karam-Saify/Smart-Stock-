#!/usr/bin/env python3
"""CLI for data ingestion, audit, and staged statistical research."""
import argparse
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from stockresearch import data, synthetic
from stockresearch.quality import audit_panels
from stockresearch.experiments import run_experiments, TEST_IDS
from stockresearch.time_series import run_time_series_experiments, TEST_IDS as TIME_SERIES_TEST_IDS
from stockresearch.universe import stage_stocks
from stockresearch.config import INITIAL_SIX
from stockresearch.risk import run_risk_experiments, TEST_IDS as RISK_TEST_IDS
from stockresearch.ledger import write_combined_ledger
from stockresearch.experiment_metadata import build_experiment_metadata
from stockresearch.research_registry import list_methods


def main():
    p = argparse.ArgumentParser(description="Smart Stock Research Lab")
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("download")
    d.add_argument("--data-dir", default="data")
    d.add_argument("--start")
    d.add_argument("--end")
    d.add_argument("--synthetic", action="store_true")
    a = sub.add_parser("audit")
    a.add_argument("--data-dir", default="data")
    t = sub.add_parser("test")
    t.add_argument("--data-dir", default="data")
    t.add_argument("--output-dir", default="results")
    t.add_argument("--stage", choices=["initial", "comparative", "exploratory", "final"], default="initial")
    t.add_argument("--tests", nargs="+", choices=TEST_IDS + TIME_SERIES_TEST_IDS + RISK_TEST_IDS)
    t.add_argument("--question", default="", help="Research question recorded in the experiment manifest")
    t.add_argument("--hypothesis", default="", help="Predeclared hypothesis recorded in the experiment manifest")
    t.add_argument("--notes", default="", help="Protocol notes and assumptions recorded in the manifest")
    methods = sub.add_parser("methods", help="List implemented research methods")
    methods.add_argument("--domain", help="Filter by research domain")
    methods.add_argument("--json", action="store_true", help="Print machine-readable method metadata")
    report = sub.add_parser("report")
    report.add_argument("--output-dir", default="results")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "methods":
        import json
        catalogue = list_methods(args.domain)
        if args.json:
            print(json.dumps([{
                "test_id": item.test_id, "domain": item.domain,
                "question": item.question, "methods": list(item.methods),
                "limitations": list(item.limitations), "status": item.status,
            } for item in catalogue], indent=2))
        else:
            for item in catalogue:
                print(f"{item.test_id} [{item.domain}] {item.question}")
                print("  Methods:", ", ".join(item.methods))
                print("  Limitations:", "; ".join(item.limitations))
    elif args.command == "download":
        if args.synthetic:
            synthetic.make(args.data_dir, end=args.end or "2025-12-31")
        else:
            failures = data.download_all(args.data_dir, args.start, args.end)
            if failures:
                raise SystemExit(f"{len(failures)} tickers failed; see download_manifest.json")
    elif args.command == "audit":
        print(audit_panels(args.data_dir).to_string(index=False))
    elif args.command == "test":
        tickers = INITIAL_SIX if args.stage == "initial" else stage_stocks(args.stage)
        selected = args.tests or (TEST_IDS + TIME_SERIES_TEST_IDS + RISK_TEST_IDS)
        first = [x for x in selected if x in TEST_IDS]
        second = [x for x in selected if x in TIME_SERIES_TEST_IDS]
        third = [x for x in selected if x in RISK_TEST_IDS]
        frames = []
        started_at = datetime.now(timezone.utc).isoformat()
        started_clock = time.perf_counter()
        # Isolate each engine's intermediate files. Some engines append to their
        # own ledgers, so a fresh temporary directory prevents stale-run mixing.
        with tempfile.TemporaryDirectory(prefix="smart-stock-engines-") as temp:
            root = Path(temp)
            if first:
                frames.append(run_experiments(args.data_dir, root / "distribution", first, tickers)[0])
            if second:
                frames.append(run_time_series_experiments(args.data_dir, root / "time_series", second, tickers)[0])
            if third:
                frames.append(run_risk_experiments(args.data_dir, root / "risk", third, tickers)[0])
        import pandas as pd
        metadata = build_experiment_metadata(
            args.data_dir, stage=args.stage, tickers=tickers, test_ids=selected,
            research_question=args.question, hypothesis=args.hypothesis, notes=args.notes,
            started_at=started_at, elapsed_seconds=round(time.perf_counter() - started_clock, 6),
        )
        results, summary = write_combined_ledger(frames, args.output_dir, metadata)
        print(summary.to_string(index=False))
        print(f"Saved {len(results)} method-level result rows to {args.output_dir}/experiment_results.csv")
        print(f"Saved reproducibility manifest to {args.output_dir}/experiment_manifest.json")
    elif args.command == "report":
        import pandas as pd
        path = Path(args.output_dir) / "experiment_results.csv"
        if not path.exists():
            raise SystemExit(f"No results found at {path}; run 'test' first.")
        frame = pd.read_csv(path)
        if frame.empty:
            print("The result ledger is empty.")
        else:
            print(frame.groupby(["test_id", "status"]).size().to_string())
        print(f"Detailed ledger: {path}")


if __name__ == "__main__":
    main()
