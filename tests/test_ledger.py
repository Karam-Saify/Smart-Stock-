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
