"""Daily virtual paper runner. No live broker, account, or order path."""
import json
from pathlib import Path
import numpy as np, pandas as pd
from .config import TradingConfig
from .broker import PaperBroker, Order
from .protocol import load_prices, ProtocolError
from .riskgov import clip_weights
from .strategies import Candidate

def _append(path,rows):
    if rows: pd.DataFrame(rows).to_csv(path,mode="a",header=not path.exists(),index=False)

def run_paper_day(data_dir,paper_dir,out_dir,cfg=None,today=None,allow_stale=False,ack_unvalidated=False):
    cfg=cfg or TradingConfig(); pdx=Path(paper_dir); out=Path(out_dir)
    sp=out/"selection.json"
    if not sp.exists(): raise ProtocolError("No frozen selection; run evaluate first.")
    sel=json.loads(sp.read_text()); fp=out/"final_result.json"
    gated=fp.exists() and json.loads(fp.read_text()).get("eligible_for_paper",False)
    if not gated and not ack_unvalidated: raise ProtocolError("All validation and final gates must pass; override is for virtual diagnostics only.")
    px=load_prices(data_dir,cfg)
    if list(px.columns)!=sel["universe"]: raise ProtocolError("Price panel differs from frozen universe.")
    last_date=px.index[-1]; today=today or pd.Timestamp.now().normalize()
    if (today-last_date).days>cfg.max_staleness_days and not allow_stale: raise ProtocolError(f"Stale data: {last_date.date()}.")
    b=PaperBroker.load(pdx/"state.json",cfg.costs); st=b.state; run_id=str(last_date.date())
    if b.already_ran(run_id): return {"status":"already_ran","run_id":run_id}
    prices={c:float(v) for c,v in px.iloc[-1].items()}
    if not np.isfinite(list(prices.values())).all(): raise ProtocolError("Non-finite final prices.")
    eq=b.equity(prices); lim=cfg.limits; notes=[]
    if not gated and ack_unvalidated: notes.append("UNVALIDATED VIRTUAL SIMULATION OVERRIDE ACKNOWLEDGED")
    ep=pdx/"equity.csv"; prev=float(pd.read_csv(ep)["equity"].iloc[-1]) if ep.exists() else None
    halted=False
    if st.get("halt_runs_left",0)>0:
        halted=True; st["halt_runs_left"]-=1
        if st["halt_runs_left"]==0: st["peak_equity"]=eq; notes.append("cooldown over: peak reset")
    else:
        st["peak_equity"]=max(st.get("peak_equity",eq),eq); dd=1-eq/st["peak_equity"]
        day=(eq/prev-1) if prev else 0.0
        if dd>=lim.kill_drawdown:
            st["halt_runs_left"]=max(0,lim.kill_cooldown_days-1); halted=True; notes.append(f"DRAWDOWN KILL {dd:.3f}")
        elif day<=-lim.daily_loss_limit:
            st["halt_runs_left"]=max(0,lim.daily_loss_cooldown_days-1); halted=True; notes.append(f"DAILY LOSS HALT {day:.3f}")
    pending=st.get("pending"); fills=[]
    if halted or (pending and pending["decided_on"]<run_id):
        target={c:0.0 for c in px.columns} if halted else pending["weights"]
        pos=b.positions(); orders=[]
        for c in px.columns:
            sh=target.get(c,0.0)*eq/prices[c]-pos.get(c,0.0)
            small=abs(sh*prices[c])<lim.min_trade_value or (eq>0 and abs(sh*prices[c]/eq)<lim.no_trade_band)
            if halted and pos.get(c,0)>0: small=False
            if not small: orders.append(Order(c,sh,prices[c]))
        fills=b.execute(orders,run_id); st["pending"]=None
    if len(px)%cfg.rebalance_every==0 and not halted:
        cand=Candidate(**{k:sel["candidate"][k] for k in ("target_vol","trend_window","vol_window")})
        w=clip_weights(cand.weights(px.to_numpy()),lim)
        st["pending"]={"decided_on":run_id,"weights":dict(zip(px.columns,map(float,w)))}
        notes.append("targets decided for next run")
    # Persist every processed market date, including no-order days. Otherwise
    # idempotency only applies on days that happened to generate an order batch.
    b.execute([], run_id)
    b.save(); after=b.equity(prices)
    _append(ep,[{"date":run_id,"equity":after,"cash":b.cash(),"drawdown":1-after/max(st["peak_equity"],after),"halted":halted}])
    _append(pdx/"orders.csv",[{"run_id":f.run_id,"symbol":f.symbol,"shares":f.shares,"price":f.price,"cost":f.cost} for f in fills])
    with open(pdx/"audit.log","a") as f: f.write(f"{pd.Timestamp.now().isoformat()} {run_id} fills={len(fills)} halted={halted} {' | '.join(notes)}\n")
    return {"status":"ok","run_id":run_id,"equity":after,"fills":len(fills),"halted":halted,"notes":notes}

def status(paper_dir):
    b=PaperBroker.load(Path(paper_dir)/"state.json",None)
    return {"cash":b.cash(),"positions":b.positions(),"pending":b.state.get("pending"),
            "halt_runs_left":b.state.get("halt_runs_left",0),"runs":len(b.state["executed_runs"])}
