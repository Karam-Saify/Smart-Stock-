# Research dashboard

Install the optional UI dependencies and launch locally:

```bash
python -m pip install -r requirements-ui.txt
streamlit run app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

The dashboard reads results/experiment_results.csv by default. Download data with
python run.py download, or use python run.py download --synthetic for a clearly
labelled software smoke test. Use the sidebar to select a universe stage and tests,
run the existing CLI, inspect execution status and metric distributions, view
reported p-values and notes, and export filtered results.

## Confidence and uncertainty

Current engines do not consistently calculate confidence intervals, and the result
schema has no dedicated interval bounds, adjusted p-values or uncertainty metadata.
The dashboard distinguishes reported p-values from confidence intervals and shows
when intervals are absent. It does not infer confidence from a point estimate.

PBO is a split-based diagnostic, not a conventional confidence interval. Before
adding confidence bands, each experiment needs a predeclared estimand and an
uncertainty method that respects time dependence, evaluation design and
multiple-testing family. Ordinary IID bootstrap intervals are not automatically
valid for serially dependent returns.

The dashboard supports the currently implemented T01-T18 engines. T19-T40 must
be mapped from the original test register before being represented as implemented.
Market data and results remain local files. No trading signals are generated.
