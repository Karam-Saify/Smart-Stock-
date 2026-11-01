import numpy as np
import pandas as pd

from stockresearch.experiments import run_experiments


def test_t01_to_t07_run_on_synthetic_price_panel(tmp_path):
    data_dir=tmp_path/"data"
    output_dir=tmp_path/"results"
    data_dir.mkdir()
    rng=np.random.default_rng(7)
    idx=pd.bdate_range("2018-01-01",periods=800)
    returns=rng.standard_t(df=5,size=len(idx))*0.01
    prices=100*np.exp(np.cumsum(returns))
    pd.DataFrame({"AAA":prices},index=idx).to_csv(data_dir/"panel_adjclose.csv")
    rows,summary=run_experiments(data_dir,output_dir,tickers=["AAA"])
    assert set(rows.test_id)=={f"T{i:02d}" for i in range(1,8)}
    assert (rows[rows.test_id.isin(["T01","T02","T04","T05","T06","T07"])].status=="OK").all()\n    assert (rows[rows.test_id=="T03"].status.isin(["OK","ERROR"])).any()
    assert (output_dir/"experiment_results.csv").exists()
    assert (output_dir/"experiment_summary.csv").exists()


def test_absent_ticker_is_explicitly_skipped(tmp_path):
    data_dir=tmp_path/"data"
    data_dir.mkdir()
    pd.DataFrame({"AAA":[100,101,102]},index=pd.date_range("2024-01-01",periods=3)).to_csv(data_dir/"panel_adjclose.csv")
    rows,_=run_experiments(data_dir,tmp_path/"results",tests=["T01"],tickers=["MISSING"])
    assert rows.iloc[0].status=="SKIPPED"
