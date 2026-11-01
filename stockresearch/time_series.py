"""Time-series property experiments T08-T13.

All outputs are diagnostics with explicit limitations. No result is a trading signal.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, breaks_cusumolsresid
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.seasonal import seasonal_decompose
from arch.unitroot import PhillipsPerron

TEST_IDS=[f"T{i:02d}" for i in range(8,14)]
MIN_OBS=120

def _row(test,ticker,status,method,n,metrics=None,note=""):
    return {"test_id":test,"ticker":str(ticker),"status":status,"method":method,
            "observations":int(n),"metrics_json":json.dumps(metrics or {},sort_keys=True,default=float),"note":note}

def _series(prices):
    p=pd.to_numeric(prices,errors="coerce").replace([np.inf,-np.inf],np.nan)
    p=p.where(p>0).dropna()
    return p, np.log(p).diff().dropna()

def _skip(test,ticker,n,why):
    return [_row(test,ticker,"SKIPPED","",n,note=why)]

def _t08(r,ticker):
    n=len(r)
    if n<MIN_OBS:return _skip("T08",ticker,n,"Need at least 120 returns")
    out=[]
    for label,x in [("returns",r),("squared_returns",r**2)]:
        try:
            a=adfuller(x,autolag="AIC")
            k=kpss(x,regression="c",nlags="auto")
            pp=PhillipsPerron(x)
            out.append(_row("T08",ticker,"OK",label,n,{"adf_stat":float(a[0]),"adf_pvalue":float(a[1]),"adf_lags":int(a[2]),"kpss_stat":float(k[0]),"kpss_pvalue":float(k[1]),"pp_stat":float(pp.stat),"pp_pvalue":float(pp.pvalue)},"ADF/PP null: unit root; KPSS null: level stationarity. Interpret jointly; tests have different assumptions."))
        except Exception as e:out.append(_row("T08",ticker,"ERROR",label,n,note=f"{type(e).__name__}: {e}"))
    return out

def _t09(r,ticker):
    n=len(r)
    if n<MIN_OBS:return _skip("T09",ticker,n,"Need at least 120 returns")
    nlags=min(20,max(5,n//10))
    ac=acf(r,nlags=nlags,fft=True)
    pc=pacf(r,nlags=nlags,method="ywm")
    lb=acorr_ljungbox(r,lags=[5,10,20],return_df=True)
    return [_row("T09",ticker,"OK","acf_pacf_ljung_box",n,{"lags":list(range(1,nlags+1)),"acf":ac[1:].tolist(),"pacf":pc[1:].tolist(),"ljung_box":{str(i):{"stat":float(row.lb_stat),"pvalue":float(row.lb_pvalue)} for i,row in lb.iterrows()}},"Ljung-Box tests joint absence of autocorrelation through each lag; p-values are unadjusted.")]

def _t10(r,ticker):
    n=len(r)
    if n<MIN_OBS:return _skip("T10",ticker,n,"Need at least 120 returns")
    x=r.to_numpy(); y=x[1:]; lag=x[:-1]
    slope,intercept,rv,p,se=stats.linregress(lag,y)
    fitted=intercept+slope*lag
    resid=y-fitted
    half_life=float(-math.log(2)/math.log(slope)) if 0<slope<1 else None
    return [_row("T10",ticker,"OK","ar1_ou_diagnostic",n,{"ar1_phi":float(slope),"intercept":float(intercept),"pvalue_slope":float(p),"r_squared":float(rv**2),"residual_std":float(np.std(resid,ddof=2)),"conditional_half_life_observations":half_life},"Applied to demeaned returns, not price levels. AR(1) coefficient and half-life are descriptive; OU interpretation requires stronger assumptions.")]

def _t11(r,ticker):
    n=len(r)
    if n<MIN_OBS:return _skip("T11",ticker,n,"Need at least 120 returns")
    x=r.to_numpy(); out=[]
    for w in (20,60,120):
        if n<w+5:continue
        vals=[]
        for end in range(w,n+1):
            y=x[end-w:end]; t=np.arange(w)
            slope,intercept,rv,p,se=stats.linregress(t,y)
            vals.append((slope,p,rv**2))
        a=np.asarray(vals)
        out.append(_row("T11",ticker,"OK",f"rolling_linear_trend_{w}",n,{"window":w,"slope_mean":float(a[:,0].mean()),"slope_std":float(a[:,0].std()),"positive_slope_fraction":float(np.mean(a[:,0]>0)),"median_r_squared":float(np.median(a[:,2])),"nominal_p_lt_05_fraction":float(np.mean(a[:,1]<.05))},"Overlapping windows are dependent; nominal p-value fraction is descriptive, not inferential."))
    # Momentum autocorrelation across cumulative return horizons
    for h in (5,20,60):
        if n>h+10:
            f=pd.Series(x).rolling(h).sum().dropna()
            corr=float(f.autocorr(lag=h))
            out.append(_row("T11",ticker,"OK",f"cumulative_return_autocorrelation_{h}",len(f),{"lag":h,"autocorrelation":corr},"Descriptive persistence statistic; overlapping cumulative returns induce dependence."))
    return out or _skip("T11",ticker,n,"Insufficient data for selected rolling windows")

def _t12(prices,ticker):
    n=len(prices)
    if n<252:return _skip("T12",ticker,n,"Need at least 252 price observations")
    # Daily data: weekday and month effects on log returns; no claim of causal seasonality.
    r=np.log(prices).diff().dropna()
    frame=pd.DataFrame({"r":r})
    frame["weekday"]=frame.index.dayofweek
    frame["month"]=frame.index.month
    out=[]
    for col in ("weekday","month"):
        groups=[g["r"].to_numpy() for _,g in frame.groupby(col) if len(g)>=10]
        if len(groups)>=2:
            f=stats.f_oneway(*groups)
            means={str(k):float(v.mean()) for k,v in frame.groupby(col)["r"]}
            out.append(_row("T12",ticker,f"OK",f"{col}_anova",len(r),{"groups":len(groups),"f_stat":float(f.statistic),"pvalue":float(f.pvalue),"group_means":means},"Unadjusted exploratory ANOVA; calendar effects can be unstable and affected by market closures and multiple testing."))
    try:
        # Weekly seasonal decomposition (5 observations) on daily return series.
        d=seasonal_decompose(r,model="additive",period=5,extrapolate_trend="freq")
        out.append(_row("T12",ticker,"OK","weekday_period_5_decomposition",len(r),{"seasonal_std":float(d.seasonal.std()),"resid_std":float(d.resid.std())},"Descriptive decomposition; period 5 approximates trading weekdays and is not a forecast test."))
    except Exception as e:out.append(_row("T12",ticker,"ERROR","weekday_period_5_decomposition",len(r),note=f"{type(e).__name__}: {e}"))
    return out

def _t13(r,ticker):
    n=len(r)
    if n<MIN_OBS:return _skip("T13",ticker,n,"Need at least 120 returns")
    x=r.to_numpy(); out=[]
    try:
        # CUSUM of OLS residuals around a constant mean.
        lm=breaks_cusumolsresid(x,ddof=1)
        out.append(_row("T13",ticker,"OK","ols_cusum",n,{"statistic":float(lm[0]),"pvalue":float(lm[1]),"critical_values":{str(k):float(v) for k,v in lm[2]}},"Tests parameter stability under its regression assumptions."))
    except Exception as e:out.append(_row("T13",ticker,"ERROR","ols_cusum",n,note=f"{type(e).__name__}: {e}"))
    try:
        import ruptures as rpt
        # PELT with BIC-like penalty on mean shifts; exploratory segmentation.
        signal=x.reshape(-1,1)
        penalty=math.log(n)*float(np.var(x))*2
        bkps=rpt.Pelt(model="l2",min_size=20,jump=5).fit(signal).predict(pen=penalty)
        points=[int(v) for v in bkps[:-1]]
        out.append(_row("T13",ticker,"OK","pelt_mean_shift",n,{"break_indices":points,"break_count":len(points),"penalty":penalty},"Data-driven exploratory break locations; no calibrated p-values. Penalty sensitivity and out-of-sample stability are required."))
    except Exception as e:out.append(_row("T13",ticker,"ERROR","pelt_mean_shift",n,note=f"{type(e).__name__}: {e}"))
    return out

_FUNCS={"T08":lambda p,r,t:_t08(r,t),"T09":lambda p,r,t:_t09(r,t),"T10":lambda p,r,t:_t10(r,t),"T11":lambda p,r,t:_t11(r,t),"T12":lambda p,r,t:_t12(p,t),"T13":lambda p,r,t:_t13(r,t)}

def run_time_series_experiments(data_dir="data",output_dir="results",tests=None,tickers=None):
    path=Path(data_dir)/"panel_adjclose.csv"
    if not path.exists():raise FileNotFoundError(f"Adjusted-price panel not found: {path}")
    panel=pd.read_csv(path,index_col=0,parse_dates=True).sort_index()
    selected=list(tickers or panel.columns); chosen=list(tests or TEST_IDS)
    unknown=sorted(set(chosen)-set(TEST_IDS))
    if unknown:raise ValueError(f"Unsupported tests: {unknown}; implemented: {TEST_IDS}")
    rows=[]
    for ticker in selected:
        if ticker not in panel:
            rows.extend(_row(t,ticker,"SKIPPED","not_in_panel",0,note="Ticker absent from panel") for t in chosen);continue
        prices,r=_series(panel[ticker])
        for test in chosen:
            try:rows.extend(_FUNCS[test](prices,r,ticker))
            except Exception as e:rows.append(_row(test,ticker,"ERROR","execution",len(r),note=f"{type(e).__name__}: {e}"))
    frame=pd.DataFrame(rows,columns=["test_id","ticker","status","method","observations","metrics_json","note"])
    dest=Path(output_dir);dest.mkdir(parents=True,exist_ok=True)
    file=dest/"time_series_results.csv"
    if file.exists():
        old=pd.read_csv(file); frame=pd.concat([old,frame],ignore_index=True)
    frame.to_csv(file,index=False)
    return frame,frame.groupby(["test_id","status"]).size().unstack(fill_value=0).reset_index()
