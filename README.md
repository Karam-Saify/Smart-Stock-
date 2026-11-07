# Smart Stock Research Lab

A reproducible, hypothesis-driven environment for statistical research on equity and ETF time series. It is an empirical research tool, not an investment-advice or trading-signal service.

## Capabilities
- 48-instrument research universe: 36 stocks, 4 broad-market ETFs, 8 sector ETFs.
- T01–T40 experiment register, staged research, and long-form result ledger.
- Discovery / validation / locked-final chronological evaluation.
- Raw data retention, panel construction, quality metadata, and offline synthetic data.
- Explicit ERROR and SKIPPED rows; no silent test failures.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run.py download
python run.py test --stage initial --tests T01 T02 T03 T04 T05 T06 T07
python run.py report
python -m pytest -q
```

Offline smoke test: `python run.py download --synthetic`, then run tests and report. Synthetic observations are never real market data.

The `test` command writes one combined `experiment_results.csv` and
`experiment_summary.csv` across all selected engines. Each invocation starts with
fresh intermediate engine outputs, so previous runs are not silently appended to
the current run. The latest run replaces the canonical ledger.

## Research rules
1. Pre-register hypotheses, metrics, thresholds, and decision criteria before examining validation/final outcomes.
2. Keep final data locked until the protocol is frozen; run final evaluation once.
3. Use chronological splits and purge overlapping labels.
4. Distinguish statistical significance, predictive value, and economic relevance.
5. Control multiple testing and report dependence limitations.
6. Preserve raw vendor fields, data provenance, corporate actions, missingness, and failures.
7. The current-company universe is survivorship-biased; do not generalize it to the historical market.

Yahoo Finance is a convenient but unaudited data source. Verify corporate actions, ticker histories, and vendor revisions. Strategy-like experiments are hypothetical only and do not establish profitability.
