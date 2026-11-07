# Model-selection and backtest-overfitting diagnostics

`stockresearch.overfitting.probability_of_backtest_overfitting` implements a
compact combinatorially symmetric cross-validation (CSCV) diagnostic.

## Input and calculation

Supply a complete chronological DataFrame: rows are observations and columns are
already-defined candidate return series. The implementation divides observations
into equal contiguous slices, enumerates every in-sample choice of half the slices,
selects the candidate with the highest in-sample arithmetic mean, and records that
candidate's percentile rank on the complementary slices. PBO is the fraction of
splits where the selected candidate falls below the out-of-sample median.

The default is eight slices, yielding 70 complementary splits. Any remainder that
does not fill a complete slice is explicitly excluded and reported.

## Interpretation and limitations

PBO is a selection-instability diagnostic, not a forecast of future performance and
not proof that a strategy will fail or succeed. Results depend on the candidate set,
selection statistic, slice count, return horizon, and sample period. The simple
implementation uses mean return for selection; it does not account for costs,
capacity, serial dependence, overlapping labels, or nonstationarity. CSCV partitions
are not a substitute for a final untouched chronological holdout.

Do not construct the candidate set after inspecting the same evaluation results.
Document all candidates, parameter searches, exclusions, and evaluation windows.
No trading instructions or portfolio allocations are produced.
