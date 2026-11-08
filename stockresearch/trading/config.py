"""Frozen assumptions and guardrails for the research-only simulator."""
from __future__ import annotations
import hashlib, json
from dataclasses import asdict, dataclass, field
from .. import universe

@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.5
    etf_half_spread_bps: float = 1.0
    stock_half_spread_bps: float = 3.0
    slippage_bps: float = 1.0

    def fee_rate(self, ticker: str) -> float:
        half = self.etf_half_spread_bps if ticker in universe.BROAD + universe.SECTOR_ETFS else self.stock_half_spread_bps
        return (self.commission_bps + half) / 1e4

    def slippage_rate(self) -> float:
        return self.slippage_bps / 1e4

    def rate(self, ticker: str) -> float:
        return self.fee_rate(ticker) + self.slippage_rate()

@dataclass(frozen=True)
class RiskLimits:
    max_weight: float = 0.35
    max_gross: float = 1.0
    kill_drawdown: float = 0.20
    kill_cooldown_days: int = 20
    daily_loss_limit: float = 0.05
    daily_loss_cooldown_days: int = 5
    no_trade_band: float = 0.01
    min_trade_value: float = 50.0

@dataclass(frozen=True)
class Criteria:
    min_sharpe_ratio_vs_benchmark: float = 1.0
    max_drawdown_ratio_vs_benchmark: float = 0.8
    min_dsr: float = 0.95
    max_pbo: float = 0.5
    max_spa_pvalue: float = 0.10

@dataclass(frozen=True)
class TradingConfig:
    universe: tuple = tuple(universe.BROAD + universe.SECTOR_ETFS)
    benchmark: str = "SPY"
    rebalance_every: int = 5
    split: tuple = (0.6, 0.2, 0.2)
    costs: CostModel = field(default_factory=CostModel)
    limits: RiskLimits = field(default_factory=RiskLimits)
    criteria: Criteria = field(default_factory=Criteria)
    max_staleness_days: int = 4
    seed: int = 42

    def fingerprint(self, grid_names=()) -> str:
        raw = json.dumps({"cfg": asdict(self), "grid": list(grid_names)}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
