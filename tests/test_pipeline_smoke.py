"""Lightweight pipeline tests that avoid network access."""
import csv
import zipfile
import io
from pathlib import Path

import pandas as pd

from pipeline.download_data import date_range
from pipeline.parse_data import _parse_single, extract_date_from_filename


def test_extract_date_from_filename():
    assert extract_date_from_filename("BTCUSDT-aggTrades-2024-01-01.zip") == "2024-01-01"
    assert extract_date_from_filename("ETHUSDT-bookTicker-2024-06-30.zip") == "2024-06-30"


def test_date_range_is_inclusive():
    dates = [d.isoformat() for d in date_range("2024-01-01", "2024-01-03")]
    assert dates == ["2024-01-01", "2024-01-02", "2024-01-03"]


def test_parse_single_is_idempotent(tmp_path):
    zip_path = tmp_path / "BTCUSDT-aggTrades-2024-01-01.zip"
    out_path = tmp_path / "parsed" / "2024-01-01.parquet"

    rows = [
        [1, 50000.0, 0.25, 1, 1, 1704067200000, False],
        [2, 50001.0, 0.50, 2, 2, 1704067200100, True],
    ]
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerows(rows)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("BTCUSDT-aggTrades-2024-01-01.csv", csv_buffer.getvalue())

    name, error, skipped = _parse_single(str(zip_path), "aggTrades", str(out_path), False)
    assert name == zip_path.name
    assert error is None
    assert skipped is False
    assert out_path.exists()

    parsed = pd.read_parquet(out_path)
    assert list(parsed.columns) == ["timestamp_ms", "price", "qty", "is_buyer_maker"]
    assert len(parsed) == 2

    name, error, skipped = _parse_single(str(zip_path), "aggTrades", str(out_path), False)
    assert name == zip_path.name
    assert error is None
    assert skipped is True
    assert pd.read_parquet(out_path).equals(parsed)


def test_parse_book_ticker_with_header_and_event_time(tmp_path):
    zip_path = tmp_path / "BTCUSDT-bookTicker-2024-01-01.zip"
    out_path = tmp_path / "parsed" / "2024-01-01.parquet"

    rows = [
        [
            "update_id",
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
            "transaction_time",
            "event_time",
        ],
        [1, 50000.0, 1.25, 50000.5, 1.75, 1704067200000, 1704067200005],
        [2, 50000.1, 1.50, 50000.6, 1.50, 1704067200100, 1704067200105],
    ]
    csv_buffer = io.StringIO()
    writer = csv.writer(csv_buffer)
    writer.writerows(rows)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("BTCUSDT-bookTicker-2024-01-01.csv", csv_buffer.getvalue())

    name, error, skipped = _parse_single(str(zip_path), "bookTicker", str(out_path), False)
    assert name == zip_path.name
    assert error is None
    assert skipped is False

    parsed = pd.read_parquet(out_path)
    assert list(parsed.columns) == [
        "timestamp_ms",
        "best_bid_price",
        "best_bid_qty",
        "best_ask_price",
        "best_ask_qty",
    ]
    assert parsed["timestamp_ms"].tolist() == [1704067200000, 1704067200100]
