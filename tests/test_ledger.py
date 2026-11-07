import pandas as pd
import pytest

from stockresearch.ledger import write_combined_ledger


def row(test_id, ticker="AAA", status="OK"):
    return pd.DataFrame([{
        "test_id": test_id, "ticker": ticker, "status": status,
        "method": "fixture", "observations": 100, "metrics_json": "{}", "note": ""
    }])


def test_combines_engines_without_overwriting(tmp_path):
    results, summary = write_combined_ledger([row("T01"), row("T08"), row("T14", status="SKIPPED")], tmp_path)
    assert set(results["test_id"]) == {"T01", "T08", "T14"}
    assert len(pd.read_csv(tmp_path / "experiment_results.csv")) == 3
    assert set(summary["test_id"]) == {"T01", "T08", "T14"}
    assert summary.loc[summary["test_id"] == "T14", "SKIPPED"].iloc[0] == 1


def test_rejects_incomplete_engine_frame(tmp_path):
    with pytest.raises(ValueError):
        write_combined_ledger([pd.DataFrame({"test_id": ["T01"]})], tmp_path)


def test_persists_uncertainty_rows_and_manifest_count(tmp_path):
    uncertainty = pd.DataFrame([{
        "ticker": "AAA", "estimand": "mean daily log return",
        "ci_lower": -0.01, "ci_upper": 0.02, "status": "OK",
    }])
    write_combined_ledger(
        [row("T01")], tmp_path,
        {"experiment_id": "uncertainty-fixture"}, uncertainty_results=uncertainty,
    )
    saved = pd.read_csv(tmp_path / "experiment_uncertainty.csv")
    import json
    manifest = json.loads((tmp_path / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert len(saved) == 1
    assert manifest["uncertainty_rows"] == 1
