"""Descriptive performance summaries for historical and simulated return series."""
import numpy as np
ANN = 252

def sharpe(r):
    s = r.std(ddof=1)
    return float(r.mean() / s * np.sqrt(ANN)) if np.isfinite(s) and s > 0 else 0.0

def max_drawdown(r):
    eq = (1 + r).cumprod()
    return float((1 - eq / eq.cummax()).max()) if len(eq) else 0.0

def perf_stats(r):
    r = r.dropna()
    n = len(r)
    if n < 2:
        return {"n": n}
    down = r[r < 0].std(ddof=1)
    cagr = float((1 + r).prod() ** (ANN / n) - 1)
    dd = max_drawdown(r)
    return {"n": n, "cagr": cagr, "ann_vol": float(r.std(ddof=1)*np.sqrt(ANN)),
            "sharpe": sharpe(r), "sortino": float(r.mean()/down*np.sqrt(ANN)) if down and down > 0 else 0.0,
            "max_drawdown": dd, "calmar": float(cagr/dd) if dd > 0 else 0.0,
            "total_return": float((1+r).prod()-1)}
