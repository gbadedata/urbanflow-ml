"""Tests for TLC source discovery.

These run offline: the range logic is the part that silently produces a wrong
corpus, so it is tested without touching the network.
"""

from __future__ import annotations

import pytest

from urbanflow.ingest.sources import SourceFile, month_range, parse_month, source_files


def test_parse_month() -> None:
    assert parse_month("2023-01") == (2023, 1)
    assert parse_month("2024-12") == (2024, 12)


def test_month_range_single_month() -> None:
    assert month_range("2023-01", "2023-01") == ["2023-01"]


def test_month_range_within_year() -> None:
    assert month_range("2023-01", "2023-04") == ["2023-01", "2023-02", "2023-03", "2023-04"]


def test_month_range_crosses_year_boundary() -> None:
    """December to January is where naive month arithmetic breaks."""
    assert month_range("2023-11", "2024-02") == ["2023-11", "2023-12", "2024-01", "2024-02"]


def test_month_range_spans_multiple_years() -> None:
    months = month_range("2023-01", "2025-06")
    assert len(months) == 30
    assert months[0] == "2023-01"
    assert months[-1] == "2025-06"
    assert "2024-01" in months


def test_month_range_rejects_inverted_range() -> None:
    """An inverted range should fail loudly, not return an empty corpus."""
    with pytest.raises(ValueError, match="is after"):
        month_range("2024-06", "2023-01")


def test_source_file_naming() -> None:
    source = SourceFile(service="yellow", year=2023, month=7)
    assert source.filename == "yellow_tripdata_2023-07.parquet"
    assert source.url.endswith("/trip-data/yellow_tripdata_2023-07.parquet")
    assert source.url.startswith("https://")


def test_source_file_zero_pads_month() -> None:
    source = SourceFile(service="yellow", year=2023, month=1)
    assert "2023-01" in source.filename
    assert "2023-1." not in source.filename


def test_source_files_covers_window() -> None:
    files = source_files("yellow", "2023-01", "2023-03")
    assert [f.month_str for f in files] == ["2023-01", "2023-02", "2023-03"]
    assert all(f.service == "yellow" for f in files)
