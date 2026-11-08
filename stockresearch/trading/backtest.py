"""Close-price daily backtest: one-day signal lag and one-way all-in costs."""
from dataclasses import dataclass
import numpy as np, pandas as pd
from .riskgov import RiskGovernor, clip_weights

@dataclass
class BacktestResult:
    returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series
    events: list

def run_backtest(prices, strategy, cfg, governed=True):
    px = prices.to_numpy(float); days, n = px.shape
    daily = np.zeros_like(px); daily[1:] = px[1:] / px[:-1] - 1
    daily = np.nan_to_num(daily, nan=0.0, posinf=0.0, neginf=0.0)
    rates = np.array([cfg.costs.rate(c) for c in prices.columns])
    gov = RiskGovernor(cfg.limits); w = np.zeros(n); equity = 1.0
    rets=np.zeros(days); turns=np.zeros(days); costs=np.zeros(days); weights=np.zeros((days,n))
    for d in range(days):
        gross = float(w @ daily[d]) if d else 0.0
        equity *= 1 + gross
        drift = w * (1 + daily[d]) / (1 + gross) if d and 1 + gross > 0 else w
        flat = gov.update(d, equity, gross) if governed else False
        if d % cfg.rebalance_every == 0 or flat:
            raw = strategy.weights(px[:d])
            target = np.zeros(n) if flat else (clip_weights(raw, cfg.limits) if getattr(strategy,"apply_limits",True) else raw)
            delta = target - drift
            keep = (np.abs(delta) < cfg.limits.no_trade_band) & ~(flat & (drift > 0))
            new = np.where(keep, drift, target)
            turn = np.abs(new - drift); cost = float(turn @ rates)
            equity *= 1 - cost; turns[d] = turn.sum(); w = new
        else:
            cost = 0.0; w = drift
        rets[d] = (1 + gross) * (1 - cost) - 1
        costs[d] = cost; weights[d] = w
    idx=prices.index
    return BacktestResult(pd.Series(rets,index=idx),pd.DataFrame(weights,index=idx,columns=prices.columns),
                          pd.Series(turns,index=idx),pd.Series(costs,index=idx),gov.events)
