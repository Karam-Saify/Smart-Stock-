#!/usr/bin/env python3
"""CLI for data ingestion, audit, and staged statistical research."""
import argparse
import logging
from stockresearch import data, synthetic
from stockresearch.quality import audit_panels
from stockresearch.experiments import run_experiments, TEST_IDS
from stockresearch.universe import stage_stocks


def main():
    p=argparse.ArgumentParser(description="Smart Stock Research Lab")
    sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("download"); d.add_argument("--data-dir",default="data"); d.add_argument("--start"); d.add_argument("--end"); d.add_argument("--synthetic",action="store_true")
    a=sub.add_parser("audit"); a.add_argument("--data-dir",default="data")
    t=sub.add_parser("test"); t.add_argument("--data-dir",default="data"); t.add_argument("--output-dir",default="results"); t.add_argument("--stage",choices=["initial","comparative","exploratory","final"],default="initial"); t.add_argument("--tests",nargs="+",choices=TEST_IDS)
    report=sub.add_parser("report"); report.add_argument("--output-dir",default="results")
    args=p.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    if args.command=="download":
        if args.synthetic: synthetic.make(args.data_dir,end=args.end or "2025-12-31")
        else:
            failures=data.download_all(args.data_dir,args.start,args.end)
            if failures: raise SystemExit(f"{len(failures)} tickers failed; see download_manifest.json")
    elif args.command=="audit":
        print(audit_panels(args.data_dir).to_string(index=False))
    elif args.command=="test":
        tickers=stage_stocks(args.stage)
        results,summary=run_experiments(args.data_dir,args.output_dir,args.tests,tickers)
        print(summary.to_string(index=False))
        print(f"Saved {len(results)} method-level result rows to {args.output_dir}/experiment_results.csv")
    elif args.command=="report":
        import pandas as pd
        path=f"{args.output_dir}/experiment_results.csv"
        if not __import__("pathlib").Path(path).exists(): raise SystemExit(f"No results found at {path}; run 'test' first.")
        frame=pd.read_csv(path)
        print(frame.groupby(["test_id","status"]).size().to_string())
        print(f"Detailed ledger: {path}")


if __name__=="__main__":
    main()
