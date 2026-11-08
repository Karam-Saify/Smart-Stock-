import json
import numpy as np
import pandas as pd
import pytest
from stockresearch import synthetic
from stockresearch.trading.config import TradingConfig, RiskLimits
from stockresearch.trading.strategies import Candidate, BuyHold, default_grid
from stockresearch.trading.riskgov import clip_weights, RiskGovernor
from stockresearch.trading.backtest import run_backtest
from stockresearch.trading.broker import PaperBroker, Order
from stockresearch.trading.protocol import evaluate, ProtocolError, deflated_sharpe
from stockresearch.trading import paper

@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d=tmp_path_factory.mktemp("trading-data"); synthetic.make(d,end="2024-12-31"); return d

def prices(n=400,cols=("SPY","QQQ","XLK"),seed=0):
    rng=np.random.default_rng(seed); ix=pd.bdate_range("2020-01-01",periods=n)
    return pd.DataFrame(100*np.exp(np.cumsum(rng.normal(.0004,.01,(n,len(cols))),axis=0)),ix,list(cols))

def test_backtest_costs_benchmark_and_lag():
    px=prices(); cfg=TradingConfig(); r=run_backtest(px,BuyHold(0),cfg,governed=False)
    entry=np.flatnonzero(r.weights["SPY"].to_numpy()>0)[0]
    assert r.costs.sum()==pytest.approx(cfg.costs.rate("SPY"))
    assert np.allclose(r.returns.iloc[entry+1:].to_numpy(),px["SPY"].pct_change().fillna(0).iloc[entry+1:].to_numpy())

def test_no_lookahead():
    px=prices(); k=300; a=run_backtest(px,Candidate(.1,50,40),TradingConfig()).returns
    bad=px.copy(); bad.iloc[k:]*=np.random.default_rng(1).uniform(.2,5,bad.iloc[k:].shape)
    b=run_backtest(bad,Candidate(.1,50,40),TradingConfig()).returns
    assert np.allclose(a.iloc[:k],b.iloc[:k])

def test_weights_are_long_only_and_capped():
    w=clip_weights(Candidate(.12,None,40).weights(prices(500,("SPY","QQQ","XLK","XLF")).to_numpy()),RiskLimits())
    assert (w>=0).all() and w.max()<=.35+1e-12 and w.sum()<=1+1e-12

def test_risk_governor_cooldown():
    g=RiskGovernor(RiskLimits(kill_drawdown=.2,kill_cooldown_days=3,daily_loss_limit=.5))
    assert not g.update(0,1) and not g.update(1,1.1)
    assert g.update(2,.85) and g.update(3,.9) and g.update(4,.95)
    assert not g.update(5,.95)

def test_dsr_accounts_for_trial_count():
    a,_=deflated_sharpe(.05,750,0,3,np.random.default_rng(0).normal(0,.01,5))
    b,_=deflated_sharpe(.05,750,0,3,np.random.default_rng(0).normal(0,.01,500))
    assert b<a

def test_paper_broker_is_idempotent_and_never_shorts(tmp_path):
    cfg=TradingConfig(); b=PaperBroker.init(tmp_path/"state.json",1000,cfg.costs)
    f=b.execute([Order("SPY",100,100)],"r1")
    assert f and 0<b.positions()["SPY"]<10 and b.cash()>=0
    assert b.execute([Order("SPY",100,100)],"r1")==[]
    b.execute([Order("SPY",-999,100)],"r2")
    assert b.positions()=={}

def test_slippage_is_not_double_counted(tmp_path):
    cfg=TradingConfig(); b=PaperBroker.init(tmp_path/"s.json",1000,cfg.costs)
    f=b.execute([Order("SPY",1,100)],"one")[0]
    expected=100*cfg.costs.fee_rate("SPY")+100*cfg.costs.slippage_rate()
    assert f.price==pytest.approx(100*(1+cfg.costs.slippage_rate()))
    assert f.cost==pytest.approx(expected)
    assert b.cash()==pytest.approx(1000-f.price-100*cfg.costs.fee_rate("SPY"))

def test_selection_freeze_final_one_shot_and_archive(data_dir,tmp_path):
    out=tmp_path/"out"; evaluate(data_dir,out)
    sel=json.loads((out/"selection.json").read_text())
    assert not sel["final_run"]
    with pytest.raises(ProtocolError): evaluate(data_dir,out)
    evaluate(data_dir,out,new_protocol=True)
    assert list(out.glob("protocol_v1_archive_*"))
    final=evaluate(data_dir,out,final=True)
    assert "gates" in final
    with pytest.raises(ProtocolError): evaluate(data_dir,out,final=True)

def test_paper_requires_passed_gates_and_stale_data(data_dir,tmp_path):
    out,pdir=tmp_path/"out",tmp_path/"paper"; evaluate(data_dir,out)
    PaperBroker.init(pdir/"state.json",100000,TradingConfig().costs)
    with pytest.raises(ProtocolError): paper.run_paper_day(data_dir,pdir,out)
    with pytest.raises(ProtocolError): paper.run_paper_day(data_dir,pdir,out,ack_unvalidated=True)
    result=paper.run_paper_day(data_dir,pdir,out,ack_unvalidated=True,allow_stale=True)
    assert result["status"]=="ok"
    assert paper.run_paper_day(data_dir,pdir,out,ack_unvalidated=True,allow_stale=True)["status"]=="already_ran"
