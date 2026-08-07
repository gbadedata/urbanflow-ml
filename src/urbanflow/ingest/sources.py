"""TLC source discovery.

The TLC publishes one Parquet file per service per month at a stable CloudFront
path. This module turns a (start_month, end_month) window into the concrete list
of files to retrieve, and does nothing else — no network calls live here, so the
range logic stays unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"


@dataclass(frozen=True)
class SourceFile:
    """One monthly TLC Parquet file."""

    service: str
    year: int
    month: int

    @property
    def month_str(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    @property
    def filename(self) -> str:
        return f"{self.service}_tripdata_{self.month_str}.parquet"

    @property
    def url(self) -> str:
        return f"{TLC_BASE_URL}/{self.filename}"


def parse_month(month: str) -> tuple[int, int]:
    """Parse a 'YYYY-MM' string into (year, month)."""
    year_s, month_s = month.split("-")
    return int(year_s), int(month_s)


def month_range(start_month: str, end_month: str) -> list[str]:
    """Inclusive list of 'YYYY-MM' strings from start to end.

    Raises if the range is inverted, which is otherwise a silent way to build an
    empty corpus and not notice until training.
    """
    start_year, start_m = parse_month(start_month)
    end_year, end_m = parse_month(end_month)

    start = date(start_year, start_m, 1)
    end = date(end_year, end_m, 1)
    if start > end:
        raise ValueError(f"start_month {start_month} is after end_month {end_month}")

    months: list[str] = []
    year, month = start_year, start_m
    while (year, month) <= (end_year, end_m):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            month = 1
            year += 1
    return months


def source_files(service: str, start_month: str, end_month: str) -> list[SourceFile]:
    """The list of monthly files covering the configured window."""
    files = []
    for m in month_range(start_month, end_month):
        year, month = parse_month(m)
        files.append(SourceFile(service=service, year=year, month=month))
    return files
