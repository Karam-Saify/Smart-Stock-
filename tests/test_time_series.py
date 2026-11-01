import numpy as np
import pandas as pd
from stockresearch.time_series import run_time_series_experiments


def test_t08_to_t13_execute_on_synthetic_prices(tmp_path):
    d=tmp_path/"data";d.mkdir()
    rng=np.random.default_rng(2026)
    idx=pd.bdate_range("2015-01-01",periods=1500)
    # AR(1)-like stationary return series with deterministic seed.
    e=rng.normal(0,.01,len(idx)); r=np.zeros(len(idx))
    for i in range(1,len(idx)):r[i]=.12*r[i-1]+e[i]
    p=100*np.exp(np.cumsum(r))
    pd.DataFrame({"AAA":p},index=idx).to_csv(d/"panel_adjclose.csv")
    rows,summary=run_time_series_experiments(d,tmp_path/"results",tickers=["AAA"])
    assert set(rows.test_id)=={f"T{i:02d}" for i in range(8,14)}
    assert not rows.empty
    assert (tmp_path/"results"/"time_series_results.csv").exists()


def test_missing_ticker_is_skipped(tmp_path):
    d=tmp_path/"data";d.mkdir()
    pd.DataFrame({"AAA":[100,101]},index=pd.date_range("2024-01-01",periods=2)).to_csv(d/"panel_adjclose.csv")
    rows,_=run_time_series_experiments(d,tmp_path/"results",tests=["T08"],tickers=["NOPE"])
    assert rows.iloc[0].status=="SKIPPED"
