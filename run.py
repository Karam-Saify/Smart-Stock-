#!/usr/bin/env python3
"""Command-line entry point for data ingestion and quality auditing."""
import argparse,logging
from stockresearch import data,synthetic
from stockresearch.quality import audit_panels

def main():
    p=argparse.ArgumentParser(description="Smart Stock Research Lab")
    sub=p.add_subparsers(dest="command",required=True)
    d=sub.add_parser("download"); d.add_argument("--data-dir",default="data"); d.add_argument("--start"); d.add_argument("--end"); d.add_argument("--synthetic",action="store_true")
    a=sub.add_parser("audit"); a.add_argument("--data-dir",default="data")
    args=p.parse_args(); logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    if args.command=="download":
        if args.synthetic: synthetic.make(args.data_dir,end=args.end or "2025-12-31")
        else:
            failures=data.download_all(args.data_dir,args.start,args.end)
            if failures: raise SystemExit(f"{len(failures)} tickers failed; see download_manifest.json")
    elif args.command=="audit":
        print(audit_panels(args.data_dir).to_string(index=False))
if __name__=="__main__": main()
