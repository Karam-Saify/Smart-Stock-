"""Frozen discovery/validation selection and one-shot final evaluation."""
import json, math, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from .config import TradingConfig
from .strategies import Candidate, BuyHold, default_grid
from .backtest import run_backtest
from .metrics import perf_stats, sharpe, max_drawdown
from ..overfitting import probability_of_backtest_overfitting

class ProtocolError(RuntimeError): pass

def load_prices(data_dir,cfg):
    p=pd.read_csv(Path(data_dir)/"panel_adjclose.csv",index_col=0,parse_dates=True).sort_index()
    cols=[c for c in cfg.universe if c in p.columns]
    if cfg.benchmark not in cols: raise ProtocolError(f"benchmark {cfg.benchmark} missing from panel")
    px=p[cols].dropna(how="any")
    if len(px)<1500: raise ProtocolError(f"only {len(px)} common rows; need >= 1500")
    return px

def deflated_sharpe(sr,n,skew,kurt,trial_srs):
    m=len(trial_srs)
    if m<2: return float("nan"),float("nan")
    v=float(np.var(trial_srs,ddof=1)); g=.5772156649
    sr0=math.sqrt(v)*((1-g)*stats.norm.ppf(1-1/m)+g*stats.norm.ppf(1-1/(m*math.e)))
    den=math.sqrt(max(1-skew*sr+(kurt-1)/4*sr**2,1e-9))
    return float(stats.norm.cdf((sr-sr0)*math.sqrt(n-1)/den)),float(sr0)

def sharpe_diff_ci(a,b,block=20,reps=2000,seed=17):
    x,y=a.to_numpy(),b.to_numpy(); n=len(x)
    if n<60: return {"diff":float("nan"),"lo":float("nan"),"hi":float("nan")}
    rng=np.random.default_rng(seed); nb=int(np.ceil(n/block)); out=np.empty(reps)
    def sh(z):
        sd=z.std(ddof=1); return z.mean()/sd*math.sqrt(252) if sd>0 else 0.0
    for i in range(reps):
        ix=(rng.integers(0,n-block+1,nb)[:,None]+np.arange(block)).ravel()[:n]
        out[i]=sh(x[ix])-sh(y[ix])
    return {"diff":float(sh(x)-sh(y)),"lo":float(np.quantile(out,.025)),"hi":float(np.quantile(out,.975))}

def spa_pvalue(bench,cands,reference_bench=None,reference_cands=None,block=20,reps=1000,seed=17):
    try:
        from arch.bootstrap import SPA
    except Exception as exc:
        return {"error":repr(exc)}
    rb=reference_bench if reference_bench is not None else bench
    rc=reference_cands if reference_cands is not None else cands
    sb=float(rb.std(ddof=1))
    if not np.isfinite(sb) or sb<=0: return {"error":"benchmark reference volatility invalid"}
    scaled={}
    for col in cands.columns:
        sd=float(rc[col].std(ddof=1))
        if np.isfinite(sd) and sd>0: scaled[col]=cands[col]*(sb/sd)
    if not scaled: return {"error":"no candidate with positive reference volatility"}
    model=SPA((-bench).to_numpy(),(-pd.DataFrame(scaled)).to_numpy(),block_size=block,reps=reps,bootstrap="stationary",seed=seed)
    model.compute(); pv=model.pvalues
    return {"lower":float(pv["lower"]),"consistent":float(pv["consistent"]),"upper":float(pv["upper"]),
            "volatility_scaling_window":"discovery" if reference_bench is not None else "test_sample"}

def _gates(sel,bm,extra,cfg):
    c=cfg.criteria; ss,bs=sharpe(sel),sharpe(bm); sd,bd=max_drawdown(sel),max_drawdown(bm)
    ratio=ss/bs if bs>0 else (float("inf") if ss>0 else 0.0)
    dd=sd/bd if bd>0 else float("inf")
    rows=[{"gate":"sharpe_ratio_vs_benchmark","value":ratio,"rule":f">= {c.min_sharpe_ratio_vs_benchmark}","passed":bool(ratio>=c.min_sharpe_ratio_vs_benchmark)},
          {"gate":"max_drawdown_ratio_vs_benchmark","value":dd,"rule":f"<= {c.max_drawdown_ratio_vs_benchmark}","passed":bool(dd<=c.max_drawdown_ratio_vs_benchmark)}]
    if "dsr" in extra:
        rows += [{"gate":"deflated_sharpe_probability","value":extra["dsr"],"rule":f">= {c.min_dsr}","passed":bool(np.isfinite(extra["dsr"]) and extra["dsr"]>=c.min_dsr)},
                 {"gate":"probability_of_backtest_overfitting","value":extra["pbo"],"rule":f"<= {c.max_pbo}","passed":bool(np.isfinite(extra["pbo"]) and extra["pbo"]<=c.max_pbo)}]
        spa=extra.get("spa_upper",float("nan"))
        rows.append({"gate":"spa_upper_pvalue_vs_benchmark","value":spa,"rule":f"<= {c.max_spa_pvalue}","passed":bool(np.isfinite(spa) and spa<=c.max_spa_pvalue)})
    return rows

def _boundaries(index,cfg):
    n=len(index); a=int(n*cfg.split[0]); b=int(n*(cfg.split[0]+cfg.split[1]))
    return index[a-1],index[b-1]

def evaluate(data_dir,out_dir,cfg=None,final=False,new_protocol=False):
    cfg=cfg or TradingConfig(); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    grid=default_grid(); fp=cfg.fingerprint([x.name for x in grid]); path=out/"selection.json"
    px=load_prices(data_dir,cfg)
    if final: return _final(px,out,cfg,fp)
    if path.exists() and not new_protocol:
        old=json.loads(path.read_text())
        raise ProtocolError(f"Frozen selection already exists (v{old['protocol_version']}); use --new-protocol to archive and restart.")
    version=1
    if path.exists():
        old=json.loads(path.read_text()); version=old["protocol_version"]+1
        archive=out/f"protocol_v{old['protocol_version']}_archive_{time.time_ns()}"; archive.mkdir()
        for name in ("selection.json","final_result.json","candidates.csv","validation_report.json","validation_report.md"):
            f=out/name
            if f.exists(): f.replace(archive/name)
    disc_end,val_end=_boundaries(px.index,cfg); visible=px.loc[:val_end]
    disc=visible.index<=disc_end; val=~disc
    bi=list(px.columns).index(cfg.benchmark)
    bench=run_backtest(visible,BuyHold(bi),cfg,governed=False).returns
    results={x.name:run_backtest(visible,x,cfg) for x in grid}
    R=pd.DataFrame({k:v.returns for k,v in results.items()})
    rows={}
    for k,v in results.items():
        rows[k]={**{f"disc_{a}":b for a,b in perf_stats(R.loc[disc,k]).items()},
                 **{f"val_{a}":b for a,b in perf_stats(R.loc[val,k]).items()},
                 "turnover_per_year":v.turnover.sum()/(len(visible)/252),
                 "cost_drag_per_year":v.costs.sum()/(len(visible)/252)}
    table=pd.DataFrame(rows).T; best=table["disc_sharpe"].idxmax()
    sr=R.loc[disc,best]; trials=(R.loc[disc].mean()/R.loc[disc].std(ddof=1)).to_numpy()
    dsr,sr0=deflated_sharpe(float(sr.mean()/sr.std(ddof=1)),len(sr),stats.skew(sr),stats.kurtosis(sr,fisher=False),trials)
    try:
        _,pbo=probability_of_backtest_overfitting(R.loc[disc],n_slices=8,selection_metric="sharpe")
        pbo=float(pbo["pbo"])
    except Exception: pbo=float("nan")
    spa=spa_pvalue(bench[val],R.loc[val],bench[disc],R.loc[disc])
    gates=_gates(R.loc[val,best],bench[val],{"dsr":dsr,"pbo":pbo,"spa_upper":spa.get("upper",float("nan"))},cfg)
    eligible=all(x["passed"] for x in gates)
    cand=next(x for x in grid if x.name==best)
    selection={"protocol_version":version,"frozen_at":pd.Timestamp.now().isoformat(),"config_fingerprint":fp,
               "discovery_end":str(disc_end.date()),"validation_end":str(val_end.date()),"universe":list(px.columns),
               "benchmark":cfg.benchmark,"candidate":{"name":best,**cand.__dict__},"n_trials":len(grid),
               "validation_gates":gates,"validation_eligible":bool(eligible),"final_run":False}
    path.write_text(json.dumps(selection,indent=2,default=str)); table.to_csv(out/"candidates.csv")
    report={"selected":best,"benchmark_val":perf_stats(bench[val]),"selected_val":perf_stats(R.loc[val,best]),
            "selected_disc":perf_stats(R.loc[disc,best]),"benchmark_disc":perf_stats(bench[disc]),"dsr":dsr,
            "expected_max_sharpe_daily_under_null":sr0,"pbo":{"value":pbo,"metric":"discovery Sharpe"},
            "spa":spa,"gates":gates,"eligible_for_paper_after_validation":bool(eligible),
            "sharpe_diff_val_ci":sharpe_diff_ci(R.loc[val,best],bench[val])}
    (out/"validation_report.json").write_text(json.dumps(report,indent=2,default=str))
    return report

def _final(px,out,cfg,fp):
    path=out/"selection.json"
    if not path.exists(): raise ProtocolError("No frozen selection; run evaluate first.")
    sel=json.loads(path.read_text())
    if sel["final_run"]: raise ProtocolError("Final evaluation is one-shot; use --new-protocol for a new protocol.")
    if sel["config_fingerprint"]!=fp: raise ProtocolError("Config/grid changed after selection was frozen.")
    end=pd.Timestamp(sel["validation_end"])
    if px.index[-1]<=end+pd.Timedelta(days=60): raise ProtocolError("Final window is shorter than ~60 calendar days.")
    cand=Candidate(**{k:sel["candidate"][k] for k in ("target_vol","trend_window","vol_window")})
    bi=list(px.columns).index(cfg.benchmark)
    sr=run_backtest(px,cand,cfg).returns; bm=run_backtest(px,BuyHold(bi),cfg,governed=False).returns
    mask=px.index>end; gates=_gates(sr[mask],bm[mask],{},cfg)
    final={"window":[str(px.index[mask][0].date()),str(px.index[mask][-1].date())],
           "selected_stats":perf_stats(sr[mask]),"benchmark":perf_stats(bm[mask]),"gates":gates,
           "sharpe_diff_ci":sharpe_diff_ci(sr[mask],bm[mask]),
           "eligible_for_paper":bool(sel["validation_eligible"] and all(x["passed"] for x in gates)),
           "candidate":sel["candidate"]}
    (out/"final_result.json").write_text(json.dumps(final,indent=2,default=str))
    sel["final_run"]=True; sel["final_at"]=pd.Timestamp.now().isoformat(); path.write_text(json.dumps(sel,indent=2,default=str))
    return final
