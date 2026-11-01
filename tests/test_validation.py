import numpy as np
import pandas as pd
import pytest

from stockresearch.validation import audit_price_panel, assert_research_ready


def test_clean_panel_passes_structural_gates():
    idx = pd.date_range("2024-01-01", periods=5, freq="B")
    panel = pd.DataFrame({"AAA": [10, 10.2, 10.1, 10.3, 10.4]}, index=idx)
    result = assert_research_ready(panel)
    assert result["rows"] == 5
    assert result["tickers"] == 1
    assert result["warnings"] == []


def test_duplicate_and_unsorted_dates_are_errors():
    idx = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01"])
    panel = pd.DataFrame({"AAA": [10, 11, 12]}, index=idx)
    codes = set(audit_price_panel(panel)["code"])
    assert {"duplicate_dates", "unsorted_dates"} <= codes
    with pytest.raises(ValueError, match="integrity gates"):
        assert_research_ready(panel)


def test_nonpositive_prices_are_rejected():
    panel = pd.DataFrame({"AAA": [10.0, 0.0, 11.0]}, index=pd.date_range("2024-01-01", periods=3))
    assert "nonpositive_prices" in set(audit_price_panel(panel)["code"])
    with pytest.raises(ValueError):
        assert_research_ready(panel)


def test_missingness_and_extreme_returns_are_warnings():
    panel = pd.DataFrame(
        {"AAA": [10.0, 100.0, np.nan, 101.0, 102.0]},
        index=pd.date_range("2024-01-01", periods=5),
    )
    report = audit_price_panel(panel, max_missing_fraction=0.10, extreme_log_return=0.40)
    assert {"high_missingness", "extreme_returns"} <= set(report["code"])
    assert set(report["severity"]) == {"warning"}


def test_invalid_configuration_is_rejected():
    panel = pd.DataFrame({"AAA": [10.0, 11.0]}, index=pd.date_range("2024-01-01", periods=2))
    with pytest.raises(ValueError):
        audit_price_panel(panel, max_missing_fraction=1.2)
