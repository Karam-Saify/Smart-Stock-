import pytest

from scripts.scrape_market_data import parse_tickers, validate_dates


def test_default_ticker_universe_is_populated():
    tickers = parse_tickers(None)
    assert len(tickers) == 48
    assert "AAPL" in tickers
    assert "^VIX" not in tickers  # Macro series are fetched separately.


def test_provider_symbols_are_preserved():
    assert parse_tickers("brk-b, ^vix, dx-y.nyb") == ["BRK-B", "^VIX", "DX-Y.NYB"]


def test_rejects_invalid_and_duplicate_symbols():
    with pytest.raises(Exception):
        parse_tickers("AAPL,../../secret")
    with pytest.raises(Exception):
        parse_tickers("AAPL,AAPL")


def test_date_range_accepts_exclusive_end():
    validate_dates("2020-01-01", "2020-01-02")


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("2020-01-02", "2020-01-01"),
        ("2020-01-01", "2020-01-01"),
        ("not-a-date", "2020-01-02"),
    ],
)
def test_date_range_rejects_invalid_ranges(start, end):
    with pytest.raises(Exception):
        validate_dates(start, end)
