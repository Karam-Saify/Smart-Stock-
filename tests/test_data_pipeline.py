import pandas as pd
from stockresearch import synthetic
from stockresearch.quality import audit_panels

def test_synthetic_pipeline_builds_auditable_panels(tmp_path):
    synthetic.make(tmp_path,end="2021-12-31",seed=7)
    prices=pd.read_csv(tmp_path/"panel_adjclose.csv",index_col=0,parse_dates=True)
    assert len(prices.columns)==48
    assert prices.index.is_monotonic_increasing
    audit=audit_panels(tmp_path)
    assert len(audit)==48
    assert (audit.observations>0).all()
