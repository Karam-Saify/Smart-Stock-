# Research-only strategy evaluation and virtual paper simulation

This layer is an experimental statistical evaluation harness. It is not a trading recommendation and it places no real orders.

## Protocol
1. `python trade.py evaluate` evaluates 27 pre-declared ETF candidates. It selects by discovery-period net Sharpe, checks validation gates, and freezes the candidate and configuration.
2. `python trade.py evaluate --final` performs a one-shot evaluation on the later holdout window. Re-running is refused; use `--new-protocol` to archive prior protocol outputs and begin a new counted experiment.
3. `python trade.py paper-init --cash 100000` creates a virtual balance only. `python trade.py paper-run` advances the virtual simulation using refreshed data.

## Statistical safeguards
PBO uses CSCV on discovery returns only and uses the same Sharpe objective as candidate selection. SPA compares the full candidate family against the benchmark on validation returns; volatility scaling is estimated on discovery and held fixed on validation. The conservative SPA upper p-value is used. Missing PBO/SPA results fail the corresponding gate; they are never treated as passes.

## Cost and risk assumptions
The historical simulator charges commission, estimated half-spread and slippage once per traded notional. The paper broker applies slippage once through the simulated fill price and separately charges commission plus spread; its reported fill cost includes both. Assumptions are illustrative, not observed costs. The simulator is long-only, unlevered, uses exposure caps and drawdown/daily-loss cooldowns, and refuses stale data by default.

## Hard boundary and limitations
There is no live broker adapter, account connection, real-order submission, or live deployment path. The universe is today's ETF set and has survivorship limitations. The Yahoo data feed is unaudited and not point-in-time verified. Close-price fills omit intraday liquidity, market impact, taxes, FX, withholding and partial fills. Backtest and paper results are educational diagnostics, not evidence of future performance.
