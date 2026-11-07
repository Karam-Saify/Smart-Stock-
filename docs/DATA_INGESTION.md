# Real market data ingestion

Smart Stock uses a provider adapter in `stockresearch/data.py`. The operational
entry point is `scripts/scrape_market_data.py`; it downloads vendor-supplied
daily OHLCV, adjusted close and corporate-action fields for the configured
48-instrument research universe, then builds panels and runs the existing
quality audit.

## Run locally

```bash
python -m pip install pandas numpy yfinance
python scripts/scrape_market_data.py
```

Choose a smaller sample or date range:

```bash
python scripts/scrape_market_data.py --tickers AAPL,MSFT,SPY --start 2015-01-01
python scripts/scrape_market_data.py --tickers AAPL,MSFT --start 2020-01-01 --end 2025-01-01
```

The end date follows Yahoo Finance's exclusive-end convention. By default,
a partial download returns a non-zero exit code after retaining successful
downloads and writing the diagnostic summary. Use `--allow-partial` only
when a partial universe is acceptable for the research task.

## Outputs

- `data/raw/*.csv`: vendor-level daily bars and action fields, retained per symbol.
- `data/panel_adjclose.csv`: dividend- and split-adjusted close panel.
- `data/panel_close.csv`: split-adjusted close panel.
- `data/panel_volume.csv`: reported volume panel.
- `data/download_manifest.json`: retrieval times, row counts, date coverage, file hashes and failures.
- `data/metadata.json`: per-instrument coverage and panel semantics.
- `data/quality_audit.csv`: raw-bar and adjusted-panel diagnostics.
- `data/scrape_run_summary.json`: machine-readable run outcome.

The GitHub Actions workflow supports manual runs and weekday scheduled
refreshes. It uploads the complete data directory as a workflow artifact;
it does not commit market data into Git history. Artifacts are retained for
14 days. Download the artifact from the successful workflow run before
running the research pipeline locally.

## Data limitations

Yahoo Finance through `yfinance` is a convenient third-party source, not an
audited institutional feed. It can rate-limit, revise historical observations,
change symbol coverage, or return incomplete histories. The manifest records
retrieval metadata but cannot establish point-in-time availability. The
configured universe is a present-day sample and is therefore subject to
survivorship and selection bias. Corporate-action fields and adjusted prices
must be checked before interpreting long-horizon returns. The workflow does
not provide tick data, order books, delisted securities, historical index
constituents, or a survivorship-free point-in-time universe.
