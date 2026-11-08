#!/usr/bin/env python3
"""Research and virtual paper-simulation CLI. It never places real orders."""
import argparse, json, logging
from pathlib import Path
from stockresearch.trading.config import TradingConfig
from stockresearch.trading.protocol import evaluate, ProtocolError
from stockresearch.trading.broker import PaperBroker
from stockresearch.trading import paper

def main():
    p=argparse.ArgumentParser(description="Smart Stock research-only paper simulator")
    sub=p.add_subparsers(dest="cmd",required=True)
    for name in ("evaluate","paper-init","paper-run","paper-status"):
        s=sub.add_parser(name); s.add_argument("--data-dir",default="data"); s.add_argument("--out-dir",default="trading_out"); s.add_argument("--paper-dir",default="paper")
        if name=="evaluate": s.add_argument("--final",action="store_true"); s.add_argument("--new-protocol",action="store_true")
        if name=="paper-init":
            s.add_argument("--cash",type=float,default=100000.0,help="virtual starting balance only")
            s.add_argument("--overwrite",action="store_true")
        if name=="paper-run":
            s.add_argument("--ack-unvalidated",action="store_true",help="diagnostic override for virtual simulation only")
            s.add_argument("--allow-stale",action="store_true")
    a=p.parse_args(); logging.basicConfig(level=logging.INFO)
    cfg=TradingConfig()
    try:
        if a.cmd=="evaluate":
            r=evaluate(a.data_dir,a.out_dir,cfg,a.final,a.new_protocol)
            print(json.dumps(r.get("gates",r),indent=2,default=str))
        elif a.cmd=="paper-init":
            if a.cash<=0: raise ValueError("virtual cash must be positive")
            PaperBroker.init(Path(a.paper_dir)/"state.json",a.cash,cfg.costs,a.overwrite); print("virtual paper account created")
        elif a.cmd=="paper-run":
            print(json.dumps(paper.run_paper_day(a.data_dir,a.paper_dir,a.out_dir,cfg,allow_stale=a.allow_stale,ack_unvalidated=a.ack_unvalidated),indent=2,default=str))
        else: print(json.dumps(paper.status(a.paper_dir),indent=2,default=str))
    except (ProtocolError,FileExistsError,FileNotFoundError,ValueError) as exc:
        raise SystemExit(f"REFUSED: {exc}")

if __name__=="__main__": main()
