# External research components and design provenance

This project uses external research as methodological inspiration, not as a
license to copy entire frameworks or inherit their assumptions.

## Reviewed projects

| Project | Relevant capability | Integration decision |
|---|---|---|
| [ML4T Diagnostic](https://github.com/quantopian/ml4trading) | Factor diagnostics, leakage checks, multiple-testing and backtest-overfitting diagnostics | Adopt concepts selectively; verify upstream package, API, maintenance and license before dependency adoption |
| [tsfresh](https://github.com/blue-yonder/tsfresh) | Time-series feature extraction and feature selection | Optional future discovery layer; feature selection must occur inside chronological training folds |
| [Qlib](https://github.com/microsoft/qlib) | End-to-end quantitative research infrastructure | Study selected ideas; do not replace the existing architecture |
| [purgedcv](https://github.com/landtml/purgedcv) | Combinatorial purged CV, event-label overlap removal and embargo | Independently implement and test a smaller expanding-window purged splitter first |

## Controls added in this tranche

- Holm and Bonferroni family-wise error adjustments.
- Benjamini-Hochberg and Benjamini-Yekutieli false-discovery-rate adjustments.
- Expanding-window walk-forward splits with event-label overlap purging.
- Moving-block bootstrap confidence intervals for a sample mean.

## Research cautions

1. Define the hypothesis family before adjusting p-values. Reporting only favourable tests invalidates nominal error rates.
2. Benjamini-Hochberg assumes independence or suitable positive dependence; Benjamini-Yekutieli is valid under arbitrary dependence but more conservative.
3. Purging depends on correctly specified label end times. A splitter cannot detect future leakage in upstream feature construction or preprocessing.
4. A moving-block bootstrap preserves local contiguous dependence; it does not solve structural breaks, long memory, nonstationarity or model selection.
5. These utilities improve statistical hygiene; they do not establish predictive performance, causality or economic value.
