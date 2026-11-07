# Factor diagnostics

The optional `stockresearch.factor_diagnostics` module provides two cross-sectional
research summaries:

- Date-wise Spearman rank information coefficients (IC).
- Date-wise factor quantiles with mean supplied forward returns, plus an aggregate
  quantile profile.

## Input contract

The caller supplies two DataFrames with dates as rows and assets as columns:
`factors` and `forward_returns`. The module intersects dates and assets, removes
non-finite values pairwise, and reports skipped dates. It does not construct forward
returns or verify point-in-time availability. The researcher is responsible for
ensuring every factor observation was available at the date assigned and every
forward-return window begins strictly after that date.

## Interpretation limits

IC and quantile profiles are descriptive diagnostics, not evidence of causality,
profitability, or a stable effect. Serial dependence across dates, overlapping
forward-return horizons, changing asset coverage, ties, survivorship bias, and
multiple testing can materially distort inference. Do not compare many factors or
parameterizations and report only the strongest result. Pre-register the factor
family and evaluation window; use the statistical controls and locked evaluation
protocol documented elsewhere in this repository.

No portfolio weights, trade instructions, or live execution are produced.
