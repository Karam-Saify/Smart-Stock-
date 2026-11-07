import json

import pandas as pd
import pytest

from stockresearch.inference import build_inference_table
from stockresearch.ledger import write_combined_ledger


def _results():
    return pd.DataFrame([
        {"test_id": "T06", "ticker": "AAA", "status": "OK", "method": "ks", "observations": 200,
         "metrics_json": json.dumps({"ks_pvalue": 0.01, "wasserstein_distance": 0.2}), "note": ""},
        {"test_id": "T06", "ticker": "BBB", "status": "OK", "method": "ks", "observations": 200,
         "metrics_json": json.dumps({"ks_pvalue": 0.03, "wasserstein_distance": 0.1}), "note": ""},
        {"test_id": "T06", "ticker": "CCC", "status": "OK", "method": "ks", "observations": 200,
         "metrics_json": json.dumps({"ks_pvalue": 0.20}), "note": ""},
        {"test_id": "T09", "ticker": "AAA", "status": "OK", "method": "acf", "observations": 200,
         "metrics_json": json.dumps({"ljung_box": {"5": {"pvalue": 0.02}, "10": {"pvalue": 0.04}}}), "note": ""},
        {"test_id": "T06", "ticker": "BAD", "status": "ERROR", "method": "ks", "observations": 0,
         "metrics_json": json.dumps({"ks_pvalue": 0.001}), "note": "fixture error"},
    ])


def test_adjustment_applies_across_tickers_within_same_method_metric_family():
    frame = build_inference_table(_results(), adjustment="holm")
    ks = frame[frame["metric_path"] == "ks_pvalue"].sort_values("raw_pvalue")

    assert len(ks) == 3
    assert ks["family_size"].tolist() == [3, 3, 3]
    assert ks["adjusted_pvalue"].tolist() == pytest.approx([0.03, 0.06, 0.20])
    assert "BAD" not in set(frame["ticker"])


def test_different_methods_and_nested_lags_are_separate_families():
    frame = build_inference_table(_results(), adjustment="bonferroni")
    ljung = frame[frame["test_id"] == "T09"]

    assert set(ljung["metric_path"]) == {"ljung_box.5.pvalue", "ljung_box.10.pvalue"}
    assert set(ljung["family_size"]) == {1}
    assert set(ljung["adjusted_pvalue"]) == {0.02, 0.04}


def test_none_preserves_raw_pvalues_without_claiming_adjustment():
    frame = build_inference_table(_results(), adjustment="none")
    assert frame["adjusted_pvalue"].isna().all()
    assert set(frame["adjustment"]) == {"none"}


def test_invalid_adjustment_is_rejected():
    with pytest.raises(ValueError, match="adjustment"):
        build_inference_table(_results(), adjustment="invalid")


def test_ledger_persists_inference_table_and_manifest_count(tmp_path):
    results = _results()
    inference = build_inference_table(results, adjustment="holm")
    _, _ = write_combined_ledger(
        [results], tmp_path,
        {"experiment_id": "demo", "research_protocol": {"multiple_testing": {"adjustment": "holm"}}},
        inference_results=inference,
    )
    saved = pd.read_csv(tmp_path / "experiment_inference.csv")
    manifest = json.loads((tmp_path / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert len(saved) == len(inference)
    assert manifest["inference_rows"] == len(inference)
