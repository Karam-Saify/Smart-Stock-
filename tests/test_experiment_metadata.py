import json

import pandas as pd
import pytest

from stockresearch.experiment_metadata import build_experiment_metadata
from stockresearch.ledger import write_combined_ledger


def test_experiment_manifest_fingerprints_exact_dataset_and_protocol(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    panel = data / "panel_adjclose.csv"
    panel.write_text("Date,AAA\n2024-01-01,10\n2024-01-02,11\n", encoding="utf-8")
    (data / "metadata.json").write_text(json.dumps({"source": "fixture"}), encoding="utf-8")
    (data / "download_manifest.json").write_text(json.dumps({"provider": "fixture"}), encoding="utf-8")

    manifest = build_experiment_metadata(
        data, stage="initial", tickers=["AAA"], test_ids=["T01"],
        research_question="Does the distribution change?",
        hypothesis="The two periods have different distributions.",
        notes="Exploratory synthetic fixture.",
    )

    assert manifest["schema_version"] == "1.0"
    assert len(manifest["dataset"]["sha256"]) == 64
    assert manifest["dataset"]["source_metadata"]["source"] == "fixture"
    assert manifest["dataset"]["download_manifest"]["provider"] == "fixture"
    assert manifest["research_protocol"]["question"] == "Does the distribution change?"
    assert manifest["research_protocol"]["hypothesis"].startswith("The two periods")
    assert manifest["research_protocol"]["method_ids"] == ["T01"]


def test_manifest_requires_existing_price_panel(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_experiment_metadata(tmp_path, stage="initial", tickers=[], test_ids=[])


def test_ledger_writes_manifest_and_status_counts(tmp_path):
    frame = pd.DataFrame([{
        "test_id": "T01", "ticker": "AAA", "status": "OK",
        "method": "fixture", "observations": 100, "metrics_json": "{}", "note": "",
    }])
    metadata = {"experiment_id": "fixture-id", "research_protocol": {"question": "test"}}
    results, _ = write_combined_ledger([frame], tmp_path, metadata)
    saved = json.loads((tmp_path / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert len(results) == 1
    assert saved["experiment_id"] == "fixture-id"
    assert saved["result_rows"] == 1
    assert saved["status_counts"] == {"OK": 1}
