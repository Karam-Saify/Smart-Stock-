# Research protocol

## Before testing
Record the question, null, sample, frequency, transformations, exclusions, estimator, uncertainty measure, and decision threshold. Identify every parameter selected from data and confine selection to discovery.

## Evaluation
1. Discovery: develop and select specifications.
2. Validation: assess stability without repeatedly tuning against it.
3. Final: one locked evaluation after protocol freeze.
4. Replication: repeat on a distinct period or point-in-time universe.

## Statistical controls
- Define the full family of related hypotheses before examining results.
- Report raw and adjusted p-values. Use Holm for family-wise error control or Benjamini-Hochberg for false-discovery-rate control under suitable dependence assumptions. Consider Benjamini-Yekutieli when arbitrary dependence is plausible.
- Use chronological walk-forward validation for forecasting and predictive features. Random K-fold is not an acceptable default for time-ordered market data.
- For labels with overlapping outcome windows, provide each observation's event end timestamp and purge training rows whose labels overlap the test period.
- Use block-bootstrap uncertainty for serially dependent observations and report sensitivity to block length.
- Keep all data-driven feature selection, scaling, imputation and parameter tuning inside each training fold.

## Evidence labels
- Supported: pre-registered criterion met on validation and independently replicated.
- Not supported: adequately powered evaluation rejects the claim.
- Inconclusive: insufficient sample, unstable estimate, missing data or conflicting evidence.

A p-value alone does not establish effect size, predictive utility, causality or economic value. A clean software test suite is not empirical market validation.
