# Research Method Catalogue

This catalogue describes currently implemented research engines T01–T18. It is a map of available diagnostics, not a claim that every method is confirmatory or fully validated for every dataset.

| ID | Domain | Research question | Current methods | Important limitation |
|---|---|---|---|---|
| T01 | Distribution | How do return distributions compare with common parametric families? | Normal, Student-t fits, empirical quantiles, likelihood/AIC/BIC | Fit criteria are in-sample; goodness-of-fit and parameter uncertainty require additional care. |
| T02 | Distribution / stability | How does the sample mean evolve? | Rolling/expanding mean, CUSUM screening statistic | CUSUM screen is not a calibrated formal break-test p-value. |
| T03 | Volatility | How do variance estimators perform on a chronological holdout? | Rolling variance, EWMA, training-fit GARCH diagnostic | GARCH row is a training-fit diagnostic, not a like-for-like full-holdout forecast. |
| T04 | Distribution | How does asymmetry evolve? | Rolling and full-sample skewness | Rolling estimates are noisy and overlapping. |
| T05 | Distribution | How does tail-weight evolve? | Rolling and full-sample excess kurtosis | Rolling estimates are noisy and overlapping. |
| T06 | Distribution / change | Do first- and second-half return distributions differ? | KS, Anderson-Darling, Wasserstein | Exploratory period comparison; p-values require family-level multiplicity control. |
| T07 | Data diagnostics | Which observations are flagged as unusual? | Z-score, robust Z-score, empirical percentile screens | Flags are diagnostics, not automatic deletion instructions. |
| T08 | Statistical properties | Are returns and squared returns stationary? | ADF, KPSS, Phillips-Perron | Tests have different nulls and assumptions; interpret jointly. |
| T09 | Dependence | Is there serial dependence? | ACF, PACF, Ljung-Box | Lag-wise and repeated p-values are unadjusted in the current engine. |
| T10 | Dependence | Is there AR(1)-style persistence? | AR(1) regression and conditional half-life diagnostic | Half-life interpretation requires stronger model assumptions; applied to returns. |
| T11 | Trend / dependence | How do rolling trends and cumulative-return dependence behave? | Rolling regressions and cumulative-return autocorrelation | Overlapping windows and cumulative returns induce dependence. |
| T12 | Seasonality | Are there calendar-related differences? | Weekday/month ANOVA and period-5 decomposition | Exploratory, unadjusted comparisons; calendar effects can be unstable. |
| T13 | Structural change | Is there evidence of parameter instability or mean shifts? | OLS CUSUM and PELT mean segmentation | PELT locations are exploratory; penalty sensitivity and stability checks required. |
| T14 | Volatility | How do volatility forecasts compare on a holdout? | Rolling historical variance, EWMA, bounded-origin GARCH forecasts | GARCH is evaluated at sampled holdout origins; coverage differs across methods. |
| T15 | Volatility | Are ARCH effects present and how persistent is fitted variance? | ARCH-LM and GARCH persistence | Fitted persistence is specification-dependent, not proof of future persistence. |
| T16 | Volatility | Do asymmetric volatility specifications fit differently? | EGARCH and GJR-GARCH diagnostics | Information criteria are in-sample; formal out-of-sample comparison remains necessary. |
| T17 | Risk | How do VaR/CVaR estimates behave on a holdout? | Historical, Normal and Student-t VaR/CVaR | Holdout violation counts are uncertain; CVaR is estimated from training tail. |
| T18 | Regimes / structural change | Are there latent states or volatility change points? | Two-state Gaussian HMM and PELT volatility segmentation | In-sample segmentation; state stability and chronological refitting are unproven. |

## Registry rules

- A method ID is stable and must not be reused for a different research question.
- The registry must match executable engine IDs exactly.
- Do not advertise a test as implemented merely because it exists in an original proposal or test register.
- Record exploratory status and limitations alongside method metadata.
- Add a method only with unit tests, known-truth statistical tests, documented assumptions and an evaluation plan.
- A test passing software tests is not evidence of predictive performance or empirical market validity.

## Current validation utilities

The project also provides reusable controls in `stockresearch.statistical_controls`:
- Holm, Bonferroni, Benjamini-Hochberg and Benjamini-Yekutieli p-value adjustments.
- Expanding chronological walk-forward splitting with optional event-label overlap purging.
- Moving-block bootstrap percentile intervals for a sample mean.

These controls are not automatically applied to every engine. Hypothesis families, estimands, label end times and dependence assumptions must be defined by each research design.
