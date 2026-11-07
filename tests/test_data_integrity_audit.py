import json

import pandas as pd

from stockresearch.quality import audit_panels


def _write_fixture(root):
    raw = root / "raw"
    raw.mkdir(parents=True)
    # The second row deliberately has an invalid OHLC envelope and negative volume.
    bars = pd.DataFrame(
        {
            "Open": [10.0, 12.0, 11.0],
            "High": [11.0, 11.0, 12.0],
            "Low": [9.0, 10.0, 10.0],
            "Close": [10.5, 10.5, 11.5],
            "Adj Close": [10.0, 10.0, 11.0],
            "Volume": [1000, -5, 0],
        },
        index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-15"]),
    )
    bars.to_csv(raw / "AAA.csv")
    pd.DataFrame({"AAA": [10.0, 10.0, 11.0]}, index=bars.index).to_csv(root / "panel_adjclose.csv")
    (root / "metadata.json").write_text(
        json.dumps({"instruments": {"AAA": {"status": "ok", "first": "2024-01-02", "last": "2024-01-15"}}}),
        encoding="utf-8",
    )
    (root / "download_manifest.json").write_text(
        json.dumps({"downloaded": {"AAA": {"rows": 3}}, "failures": {}}), encoding="utf-8"
    )


def test_audit_reports_raw_bar_integrity_without_repairing(tmp_path):
    _write_fixture(tmp_path)

    report = audit_panels(tmp_path).set_index("ticker")
    row = report.loc["AAA"]

    assert row["raw_file_status"] == "read"
    assert row["ohlc_inconsistent_rows"] == 1
    assert row["negative_volume_rows"] == 1
    assert row["zero_volume_rows"] == 1
    assert row["calendar_gaps_over_7_days"] == 1
    assert row["download_status"] == "downloaded"
    assert (tmp_path / "quality_audit.csv").is_file()


def test_audit_marks_missing_raw_source_file(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    pd.DataFrame({"BBB": [5.0, 6.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"])).to_csv(
        tmp_path / "panel_adjclose.csv"
    )

    report = audit_panels(tmp_path)
    assert report.loc[0, "raw_file_status"] == "missing"
    assert report.loc[0, "download_status"] == "unknown"


def test_audit_fails_clearly_when_panel_missing(tmp_path):
    import pytest

    with pytest.raises(FileNotFoundError):
        audit_panels(tmp_path)
