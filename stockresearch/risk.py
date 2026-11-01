"""Volatility and risk experiments T14-T18.

Diagnostics only. Forecast evaluation uses a chronological holdout; no trading
recommendations or strategy sizing are produced.
"""
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import het_arch

TEST_IDS=[f"T{i:02d}" for i in range(14,19)]
MIN_OBS=252

def _row(test,ticker,status,method,n,metrics=None,note=""):
    return {"test_id":test,"ticker":str(ticker),"status":status,"method":method,
            "observations":int(n),"metrics_json":json.dumps(metrics or {},sort_keys=True,default=float),"note":note}

def _skip(test,ticker,n,msg):
    return [_row(test,ticker,"SKIPPED","",n,note=msg)]

def _load_returns(prices):
    p=pd.to_numeric(prices,errors="coerce").replace([np.inf,-np.inf],np.nan)
    p=p.where(p>0).dropna()
    return np.log(p).diff().dropna()

def _var_metrics(train,test,alpha=.05):
    q=float(np.quantile(train,alpha))
    tail=train[train<=q]
    cvar=float(tail.mean()) if len(tail) else q
    hits=test<q
    return {"var_5pct":q,"cvar_5pct":cvar,"holdout_n":len(test),
            "violation_count":int(hits.sum()),"violation_rate":float(hits.mean()),
            "average_realized_beyond_var":float(test[hits].mean()) if hits.any() else None}

def _t14(r,ticker):
    n=len(r)
    if n<MIN_OBS:return _skip("T14",ticker,n,"Need at least 252 returns")
    x=r.to_numpy(); split=int(n*.7); train=x[:split]; test=x[split:]
    if len(test)<30:return _skip("T14",ticker,n,"Need at least 30 holdout returns")
    out=[]; history=list(train)
    for w in (20,60):
        forecasts=[]
        for actual in test:
            forecasts.append(float(np.var(history[-w:],ddof=1)))
            history.append(float(actual))
        f=np.maximum(np.asarray(forecasts),1e-12); y=test**2
        out.append(_row("T14",ticker,"OK",f"rolling_historical_vol_{w}",n,{"holdout_n":len(test),"mse_squared_return":float(np.mean((y-f)**2)),"qlike":float(np.mean(np.log(f)+y/f))}))
        history=list(train)
    lam=.94; v=float(np.var(train,ddof=1)); forecasts=[]
    for actual in test:
        forecasts.append(v);v=lam*v+(1-lam)*float(actual**2)
    f=np.maximum(np.asarray(forecasts),1e-12);y=test**2
    out.append(_row("T14",ticker,"OK","ewma_lambda_0.94",n,{"holdout_n":len(test),"mse_squared_return":float(np.mean((y-f)**2)),"qlike":float(np.mean(np.log(f)+y/f))}))
    try:
        from arch import arch_model
        fit=arch_model(train*100,mean="Zero",vol="GARCH",p=1,q=1,dist="t",rescale=False).fit(disp="off")
        forecasts=[]; hist=list(train)
        # Expanding refit at a bounded number of origins; genuine one-step forecasts.
        origins=np.unique(np.linspace(0,len(test)-1,min(24,len(test)),dtype=int))
        pred=np.full(len(test),np.nan)
        for pos in origins:
            sample=np.r_[train,test[:pos]]
            fitted=arch_model(sample*100,mean="Zero",vol="GARCH",p=1,q=1,dist="t",rescale=False).fit(disp="off")
            pred[pos]=float(fitted.forecast(horizon=1,reindex=False).variance.iloc[-1,0])/10000
        valid=np.isfinite(pred)
        if valid.sum()>=5:
            y=test[valid]**2;f=np.maximum(pred[valid],1e-12)
            out.append(_row("T14",ticker,"OK","garch_t_bounded_refit_origins",n,{"forecast_origins":int(valid.sum()),"mse_squared_return":float(np.mean((y-f)**2)),"qlike":float(np.mean(np.log(f)+y/f))},"GARCH compared only at sampled holdout origins; other methods have full holdout coverage."))
        else:out.append(_row("T14",ticker,"SKIPPED","garch_t_bounded_refit_origins",n,note="Too few successful forecast origins"))
    except Exception as e:out.append(_row("T14",ticker,"ERROR","garch_t_bounded_refit_origins",n,note=f"{type(e).__name__}: {e}"))
    return out

def _t15(r,ticker):
    n=len(r)
    if n<MIN_OBS:return _skip("T15",ticker,n,"Need at least 252 returns")
    x=r.to_numpy();out=[]
    try:
        lm=het_arch(x,nlags=min(10,max(1,n//20)))
        out.append(_row("T15",ticker,"OK","arch_lm",n,{"lm_stat":float(lm[0]),"lm_pvalue":float(lm[1]),"f_stat":float(lm[2]),"f_pvalue":float(lm[3])},"Null: no ARCH effects at selected lag order."))
    except Exception as e:out.append(_row("T15",ticker,"ERROR","arch_lm",n,note=f"{type(e).__name__}: {e}"))
    try:
        from arch import arch_model
        fit=arch_model(x*100,mean="Zero",vol="GARCH",p=1,q=1,dist="t",rescale=False).fit(disp="off")
        params=fit.params
        alpha=float(params.get("alpha[1]",np.nan)); beta=float(params.get("beta[1]",np.nan))
        out.append(_row("T15",ticker,"OK","garch_1_1_persistence",n,{"omega":float(params.get("omega",np.nan)),"alpha":alpha,"beta":beta,"persistence":alpha+beta,"aic":float(fit.aic),"bic":float(fit.bic),"convergence_flag":int(fit.convergence_flag)},"Persistence near one indicates highly persistent conditional variance under this fitted specification."))
    except Exception as e:out.append(_row("T15",ticker,"ERROR","garch_1_1_persistence",n,note=f"{type(e).__name__}: {e}"))
    return out

def _t16(r,ticker):
    n=len(r)
    if n<MIN_OBS:return _skip("T16",ticker,n,"Need at least 252 returns")
    from arch import arch_model
    out=[]
    for vol in ("EGARCH","GARCH"):
        try:
            fit=arch_model(r.to_numpy()*100,mean="Zero",vol=vol,p=1,o=1,q=1,dist="t",rescale=False).fit(disp="off")
            params={str(k):float(v) for k,v in fit.params.items()}
            out.append(_row("T16",ticker,"OK",f"{vol.lower()}_asymmetry",n,{"parameters":params,"aic":float(fit.aic),"bic":float(fit.bic),"convergence_flag":int(fit.convergence_flag)},"GJR-GARCH uses the leverage term gamma[1]; EGARCH asymmetry is represented by its gamma parameter. Compare information criteria cautiously; formal out-of-sample comparison is still needed."))
        except Exception as e:out.append(_row("T16",ticker,"ERROR",f"{vol.lower()}_asymmetry",n,note=f"{type(e).__name__}: {e}"))
    return out

def _t17(r,ticker):
    n=len(r)
    if n<MIN_OBS:return _skip("T17",ticker,n,"Need at least 252 returns")
    x=r.to_numpy();split=int(n*.7);train=x[:split];test=x[split:]
    if len(test)<30:return _skip("T17",ticker,n,"Need at least 30 holdout observations")
    out=[]
    qn=float(np.quantile(train,.05));tn=float(np.mean(train))
    sd=float(np.std(train,ddof=1))
    qnorm=tn+sd*stats.norm.ppf(.05)
    df,loc,scale=stats.t.fit(train)
    qt=float(stats.t.ppf(.05,df,loc,scale))
    for method,q in (("historical",qn),("normal",qnorm),("student_t",qt)):
        tail=train[train<=q];cvar=float(tail.mean()) if len(tail) else q
        hits=test<q
        out.append(_row("T17",ticker,"OK",f"{method}_var_cvar",n,{"alpha":.05,"var":q,"cvar":cvar,"holdout_n":len(test),"violation_count":int(hits.sum()),"violation_rate":float(hits.mean()),"mean_exceedance":float(test[hits].mean()) if hits.any() else None},"Thresholds estimated on first 70%; evaluated on final 30%. VaR coverage uncertainty is substantial for short holdouts; CVaR estimate is in-sample training tail mean."))
    return out

def _t18(r,ticker):
    n=len(r)
    if n<MIN_OBS:return _skip("T18",ticker,n,"Need at least 252 returns")
    x=r.to_numpy();out=[]
    # Features use only contemporaneous and lagged returns; no future observations.
    feat=pd.DataFrame({"abs_return":np.abs(x),"squared_return":x*x})
    feat["rolling_vol"]=pd.Series(x).rolling(20,min_periods=10).std().fillna(pd.Series(x).expanding(min_periods=2).std()).fillna(0.0)
    z=(feat-feat.mean())/(feat.std(ddof=1).replace(0,1))
    try:
        from hmmlearn.hmm import GaussianHMM
        model=GaussianHMM(n_components=2,covariance_type="diag",n_iter=300,random_state=17,min_covar=1e-5)
        model.fit(z.to_numpy())
        states=model.predict(z.to_numpy())
        vol_by_state={str(int(s)):float(np.mean(np.abs(x[states==s]))) for s in np.unique(states)}
        out.append(_row("T18",ticker,"OK","two_state_gaussian_hmm",n,{"state_counts":{str(int(s)):int(np.sum(states==s)) for s in np.unique(states)},"mean_abs_return_by_state":vol_by_state,"transition_matrix":model.transmat_.tolist(),"converged":bool(model.monitor_.converged)},"In-sample latent-state segmentation; state labels are arbitrary and require chronological refit/stability checks."))
    except Exception as e:out.append(_row("T18",ticker,"ERROR","two_state_gaussian_hmm",n,note=f"{type(e).__name__}: {e}"))
    try:
        import ruptures as rpt
        signal=feat["rolling_vol"].to_numpy().reshape(-1,1)
        pen=math.log(n)*float(np.var(signal))*2
        bkps=rpt.Pelt(model="l2",min_size=20,jump=5).fit(signal).predict(pen=pen)
        out.append(_row("T18",ticker,"OK","pelt_volatility_change_points",n,{"break_indices":[int(v) for v in bkps[:-1]],"break_count":len(bkps)-1,"penalty":pen},"Exploratory change points, not probabilistic regime assignments; penalty sensitivity is required."))
    except Exception as e:out.append(_row("T18",ticker,"ERROR","pelt_volatility_change_points",n,note=f"{type(e).__name__}: {e}"))
    return out

_FUNCS={"T14":_t14,"T15":_t15,"T16":_t16,"T17":_t17,"T18":_t18}

def run_risk_experiments(data_dir="data",output_dir="results",tests=None,tickers=None):
    path=Path(data_dir)/"panel_adjclose.csv"
    if not path.exists():raise FileNotFoundError(f"Adjusted-price panel not found: {path}")
    panel=pd.read_csv(path,index_col=0,parse_dates=True).sort_index()
    selected=list(tickers or panel.columns);chosen=list(tests or TEST_IDS)
    unknown=sorted(set(chosen)-set(TEST_IDS))
    if unknown:raise ValueError(f"Unsupported tests: {unknown}; implemented: {TEST_IDS}")
    rows=[]
    for ticker in selected:
        if ticker not in panel:
            rows.extend(_row(t,ticker,"SKIPPED","not_in_panel",0,note="Ticker absent from panel") for t in chosen);continue
        r=_load_returns(panel[ticker])
        for test in chosen:
            try:rows.extend(_FUNCS[test](r,ticker))
            except Exception as e:rows.append(_row(test,ticker,"ERROR","execution",len(r),note=f"{type(e).__name__}: {e}"))
    frame=pd.DataFrame(rows,columns=["test_id","ticker","status","method","observations","metrics_json","note"])
    dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True);file=dest/"risk_results.csv"
    if file.exists():frame=pd.concat([pd.read_csv(file),frame],ignore_index=True)
    frame.to_csv(file,index=False)
    return frame,frame.groupby(["test_id","status"]).size().unstack(fill_value=0).reset_index()
