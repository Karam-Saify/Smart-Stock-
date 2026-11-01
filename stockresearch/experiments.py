"""First-pass implementations for T01-T07.

These are descriptive/statistical diagnostics, not trading signals. Every output
records its method and status; missing/short samples are SKIPPED, exceptions ERROR.
"""
from __future__ import annotations

import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


TEST_IDS = [f"T{i:02d}" for i in range(1, 8)]
MIN_OBS = 60


def _clean_returns(prices: pd.Series) -> pd.Series:
    p = pd.to_numeric(prices, errors="coerce").replace([np.inf, -np.inf], np.nan)
    p = p.where(p > 0).dropna()
    return np.log(p).diff().replace([np.inf, -np.inf], np.nan).dropna()


def _row(test_id, ticker, status, method, n, metrics=None, note=""):
    return {
        "test_id": test_id, "ticker": str(ticker), "status": status,
        "method": method, "observations": int(n),
        "metrics_json": json.dumps(metrics or {}, sort_keys=True, default=float),
        "note": note,
    }


def _t01(r, ticker):
    n = len(r)
    if n < MIN_OBS:
        return [_row("T01", ticker, "SKIPPED", "normal_vs_student_t", n, note="Need at least 60 returns")]
    x = r.to_numpy()
    mu, sigma = stats.norm.fit(x)
    t_df, t_loc, t_scale = stats.t.fit(x)
    norm_ll = float(np.sum(stats.norm.logpdf(x, mu, sigma)))
    t_ll = float(np.sum(stats.t.logpdf(x, t_df, t_loc, t_scale)))
    norm_k, t_k = 2, 3
    empirical = np.quantile(x, [0.01, 0.05, 0.5, 0.95, 0.99]).tolist()
    return [
        _row("T01", ticker, "OK", "normal", n, {"log_likelihood": norm_ll, "aic": 2*norm_k-2*norm_ll, "bic": math.log(n)*norm_k-2*norm_ll, "mean": mu, "scale": sigma}),
        _row("T01", ticker, "OK", "student_t", n, {"log_likelihood": t_ll, "aic": 2*t_k-2*t_ll, "bic": math.log(n)*t_k-2*t_ll, "df": t_df, "location": t_loc, "scale": t_scale}),
        _row("T01", ticker, "OK", "empirical_quantiles", n, {"quantile_levels": [0.01,0.05,0.5,0.95,0.99], "quantiles": empirical}, "Non-parametric reference; no likelihood comparison"),
    ]


def _t02(r, ticker):
    n=len(r)
    if n < MIN_OBS: return [_row("T02",ticker,"SKIPPED","rolling_expanding_cusum",n,note="Need at least 60 returns")]
    x=r.to_numpy(); w=min(60,max(20,n//5))
    roll=pd.Series(x).rolling(w).mean().dropna()
    first=float(np.mean(x[:n//2])); second=float(np.mean(x[n//2:]))
    sd=float(np.std(x,ddof=1))
    centered=x-float(np.mean(x))
    path=np.cumsum(centered)/(sd*math.sqrt(n)) if sd>0 else np.zeros(n)
    return [
        _row("T02",ticker,"OK","rolling_mean",n,{"window":w,"rolling_mean_std":float(roll.std()),"rolling_mean_min":float(roll.min()),"rolling_mean_max":float(roll.max())}),
        _row("T02",ticker,"OK","expanding_mean",n,{"full_sample_mean":float(np.mean(x)),"first_half_mean":first,"second_half_mean":second,"half_difference":second-first}),
        _row("T02",ticker,"OK","cusum_screen",n,{"max_abs_standardized_cusum":float(np.max(np.abs(path))),"threshold_approx_95":1.36},"Screening statistic only; not a calibrated break-test p-value"),
    ]


def _t03(r,ticker):
    n=len(r)
    if n<MIN_OBS: return [_row("T03",ticker,"SKIPPED","rolling_ewma_garch",n,note="Need at least 60 returns")]
    x=r.to_numpy(); split=max(40,int(n*.7)); train=x[:split]; test=x[split:]
    if len(test)<10: return [_row("T03",ticker,"SKIPPED","rolling_ewma_garch",n,note="Insufficient holdout observations")]
    target=test**2
    out=[]
    for window in (20,60):
        forecasts=[]
        hist=list(train)
        for actual in test:
            forecasts.append(float(np.var(hist[-window:],ddof=1)) if len(hist)>=2 else float(np.var(train,ddof=1)))
            hist.append(float(actual))
        f=np.maximum(np.asarray(forecasts),1e-12)
        out.append(_row("T03",ticker,"OK",f"rolling_variance_{window}",n,{"holdout_n":len(test),"mse_squared_return":float(np.mean((target-f)**2)),"qlike":float(np.mean(np.log(f)+target/f))}))
    lam=.94; var=float(np.var(train,ddof=1)); forecasts=[]
    for actual in test:
        forecasts.append(max(var,1e-12)); var=lam*var+(1-lam)*float(actual**2)
    f=np.maximum(np.asarray(forecasts),1e-12)
    out.append(_row("T03",ticker,"OK","ewma_lambda_0.94",n,{"holdout_n":len(test),"mse_squared_return":float(np.mean((target-f)**2)),"qlike":float(np.mean(np.log(f)+target/f))}))
    try:
        from arch import arch_model
        fit=arch_model(train*100,mean="Zero",vol="GARCH",p=1,q=1,dist="t",rescale=False).fit(disp="off")
        fc=fit.forecast(horizon=1,start=None,reindex=False)
        # Recursive one-step forecasts from fixed training fit are not generated
        # for each holdout point here; keep this as an in-sample model diagnostic.
        out.append(_row("T03",ticker,"OK","garch_1_1_student_t_train_fit",len(train),{"aic":float(fit.aic),"bic":float(fit.bic),"convergence_flag":int(fit.convergence_flag)},"Training fit only; not ranked against holdout forecasts"))
    except Exception as exc:
        out.append(_row("T03",ticker,"ERROR","garch_1_1_student_t",n,note=f"{type(exc).__name__}: {exc}"))
    return out


def _t04(r,ticker):
    n=len(r)
    if n<MIN_OBS: return [_row("T04",ticker,"SKIPPED","rolling_skewness",n,note="Need at least 60 returns")]
    s=r.rolling(60).skew().dropna()
    return [_row("T04",ticker,"OK","rolling_skewness",n,{"window":60,"full_sample_skew":float(r.skew()),"rolling_skew_mean":float(s.mean()),"rolling_skew_std":float(s.std()),"rolling_skew_min":float(s.min()),"rolling_skew_max":float(s.max())})]


def _t05(r,ticker):
    n=len(r)
    if n<MIN_OBS: return [_row("T05",ticker,"SKIPPED","rolling_excess_kurtosis",n,note="Need at least 60 returns")]
    k=r.rolling(60).kurt().dropna()
    return [_row("T05",ticker,"OK","rolling_excess_kurtosis",n,{"window":60,"full_sample_excess_kurtosis":float(r.kurt()),"rolling_excess_kurtosis_mean":float(k.mean()),"rolling_excess_kurtosis_std":float(k.std()),"rolling_excess_kurtosis_max":float(k.max())})]


def _t06(r,ticker):
    n=len(r)
    if n<2*MIN_OBS: return [_row("T06",ticker,"SKIPPED","first_half_vs_second_half",n,note="Need at least 120 returns")]
    a,b=np.array_split(r.to_numpy(),2)
    ks=stats.ks_2samp(a,b,method="auto")
    ad=stats.anderson_ksamp([a,b])
    return [_row("T06",ticker,"OK","ks_and_anderson_darling",n,{"first_half_n":len(a),"second_half_n":len(b),"ks_statistic":float(ks.statistic),"ks_pvalue":float(ks.pvalue),"anderson_statistic":float(ad.statistic),"anderson_significance_level_percent":float(ad.significance_level),"wasserstein_distance":float(stats.wasserstein_distance(a,b))},"Unadjusted single-series period comparison; correct across ticker/method family before inference")]


def _t07(r,ticker):
    n=len(r)
    if n<MIN_OBS: return [_row("T07",ticker,"SKIPPED","zscore_robust_zscore_percentile",n,note="Need at least 60 returns")]
    x=r.to_numpy(); med=float(np.median(x)); mad=float(np.median(np.abs(x-med)))
    z=(x-float(np.mean(x)))/(float(np.std(x,ddof=1)) or np.nan)
    rz=(x-med)/(1.4826*mad) if mad>0 else np.full(n,np.nan)
    lo,hi=np.quantile(x,[.01,.99])
    return [_row("T07",ticker,"OK","outlier_screens",n,{"zscore_abs_gt_3_count":int(np.nansum(np.abs(z)>3)),"robust_zscore_abs_gt_3_5_count":int(np.nansum(np.abs(rz)>3.5)),"empirical_1_99_percentile_count":int(np.sum((x<lo)|(x>hi))),"lower_1pct":float(lo),"upper_99pct":float(hi)},"Outlier flags are diagnostics, not errors to delete automatically")]


_FUNCS={"T01":_t01,"T02":_t02,"T03":_t03,"T04":_t04,"T05":_t05,"T06":_t06,"T07":_t07}


def run_experiments(data_dir="data", output_dir="results", tests=None, tickers=None):
    root=Path(data_dir); panel_path=root/"panel_adjclose.csv"
    if not panel_path.exists():
        raise FileNotFoundError(f"Adjusted-price panel not found: {panel_path}. Run 'download' first.")
    panel=pd.read_csv(panel_path,index_col=0,parse_dates=True).sort_index()
    selected=list(tickers or panel.columns)
    chosen=list(tests or TEST_IDS)
    unknown=sorted(set(chosen)-set(TEST_IDS))
    if unknown: raise ValueError(f"Unsupported tests in this engine: {unknown}; implemented: {TEST_IDS}")
    rows=[]
    for ticker in selected:
        if ticker not in panel.columns:
            for test_id in chosen: rows.append(_row(test_id,ticker,"SKIPPED","not_in_panel",0,note="Ticker absent from adjusted-price panel"))
            continue
        returns=_clean_returns(panel[ticker])
        for test_id in chosen:
            try: rows.extend(_FUNCS[test_id](returns,ticker))
            except Exception as exc:
                rows.append(_row(test_id,ticker,"ERROR","execution",len(returns),note=f"{type(exc).__name__}: {exc}"))
    out=pd.DataFrame(rows,columns=["test_id","ticker","status","method","observations","metrics_json","note"])
    dest=Path(output_dir); dest.mkdir(parents=True,exist_ok=True)
    out.to_csv(dest/"experiment_results.csv",index=False)
    summary=out.groupby(["test_id","status"]).size().unstack(fill_value=0).reset_index()
    summary.to_csv(dest/"experiment_summary.csv",index=False)
    return out,summary
