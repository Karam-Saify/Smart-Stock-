import numpy as np
import pandas as pd
from stockresearch.risk import run_risk_experiments


def test_t14_to_t18_run_on_synthetic_prices(tmp_path):
    d=tmp_path/"data";d.mkdir()
    rng=np.random.default_rng(91)
    idx=pd.bdate_range("2010-01-01",periods=1600)
    # Volatility clusters through a persistent variance process.
    eps=rng.normal(size=len(idx)); variance=np.zeros(len(idx)); variance[0]=1e-4
    ret=np.zeros(len(idx))
    for i in range(1,len(idx)):
        variance[i]=1e-6+.93*variance[i-1]+.06*ret[i-1]**2
        ret[i]=np.sqrt(variance[i])*eps[i]
    prices=100*np.exp(np.cumsum(ret))
    pd.DataFrame({"AAA":prices},index=idx).to_csv(d/"panel_adjclose.csv")
    rows,_=run_risk_experiments(d,tmp_path/"results",tickers=["AAA"])
    assert set(rows.test_id)=={f"T{i:02d}" for i in range(14,19)}
    assert not rows.empty
    assert (tmp_path/"results"/"risk_results.csv").exists()


def test_missing_ticker_is_skipped(tmp_path):
    d=tmp_path/"data";d.mkdir()
    pd.DataFrame({"AAA":[100,101]},index=pd.date_range("2024-01-01",periods=2)).to_csv(d/"panel_adjclose.csv")
    rows,_=run_risk_experiments(d,tmp_path/"results",tests=["T14"],tickers=["MISSING"])
    assert rows.iloc[0].status=="SKIPPED"
