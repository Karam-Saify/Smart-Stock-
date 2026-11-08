"""Pre-declared long-only candidate rules. No signals or weights are recommendations."""
from dataclasses import dataclass
from itertools import product
import numpy as np

@dataclass(frozen=True)
class Candidate:
    target_vol: float
    trend_window: int | None
    vol_window: int

    @property
    def name(self):
        return f"tv{self.target_vol:.2f}_trend{self.trend_window or 'off'}_vol{self.vol_window}"

    def weights(self, px: np.ndarray) -> np.ndarray:
        t, n = px.shape
        if t < max(self.vol_window, self.trend_window or 0) + 1:
            return np.zeros(n)
        win = px[-(self.vol_window + 1):]
        eligible = np.isfinite(win).all(axis=0) & (win > 0).all(axis=0)
        if self.trend_window:
            hist = px[-self.trend_window:]
            ok = np.isfinite(hist).all(axis=0)
            sma = np.nanmean(np.where(np.isfinite(hist), hist, np.nan), axis=0)
            eligible &= ok & (px[-1] > sma)
        if not eligible.any():
            return np.zeros(n)
        r = win[1:, eligible] / win[:-1, eligible] - 1
        vol = np.maximum(r.std(axis=0, ddof=1), 1e-6)
        w = 1 / vol; w /= w.sum()
        cov = np.atleast_2d(np.cov(r, rowvar=False))
        pv = float(np.sqrt(max(w @ cov @ w, 0) * 252))
        out = np.zeros(n)
        out[eligible] = w * (min(1.0, self.target_vol / pv) if pv > 0 else 1.0)
        return out

@dataclass(frozen=True)
class BuyHold:
    index: int
    apply_limits: bool = False
    name: str = "benchmark_buy_hold"
    def weights(self, px):
        w = np.zeros(px.shape[1])
        if len(px) and np.isfinite(px[-1, self.index]):
            w[self.index] = 1.0
        return w

def default_grid():
    return [Candidate(v, t, w) for v, t, w in product((.08,.10,.12),(None,100,200),(40,60,120))]
