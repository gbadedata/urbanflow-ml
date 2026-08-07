"""Tests for snapshot provenance.

The manifest is what makes a result traceable to specific bytes (spec §6.1), so
the properties tested here are the ones that would quietly undermine that:
determinism of the snapshot ID, sensitivity to content change, and detection of
schema drift across a window.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from urbanflow.ingest.manifest import (
    FileRecord,
    SnapshotManifest,
    schema_fingerprint,
    sha256_file,
)


def _record(filename: str, sha: str, fingerprint: str = "aaaa1111") -> FileRecord:
    return FileRecord(
        filename=filename,
        url=f"https://example.invalid/{filename}",
        sha256=sha,
        bytes=1024,
        rows=1000,
        schema_fingerprint=fingerprint,
        retrieved_at="2026-01-01T00:00:00+00:00",
    )


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"urbanflow")
    digest = sha256_file(path)
    assert len(digest) == 64
    assert digest == sha256_file(path), "checksum must be stable across reads"


def test_sha256_detects_content_change(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"before")
    first = sha256_file(path)
    path.write_bytes(b"after")
    assert sha256_file(path) != first


def test_schema_fingerprint_is_order_independent() -> None:
    """Column reordering upstream is harmless and must not read as drift."""
    a = schema_fingerprint([("pickup", "timestamp[us]"), ("zone", "int64")])
    b = schema_fingerprint([("zone", "int64"), ("pickup", "timestamp[us]")])
    assert a == b


def test_schema_fingerprint_detects_type_change() -> None:
    a = schema_fingerprint([("zone", "int64")])
    b = schema_fingerprint([("zone", "int32")])
    assert a != b


def test_schema_fingerprint_detects_added_column() -> None:
    a = schema_fingerprint([("zone", "int64")])
    b = schema_fingerprint([("zone", "int64"), ("vendor", "int64")])
    assert a != b


def test_snapshot_id_is_deterministic() -> None:
    """Two runs over identical content must yield the same ID."""
    first = SnapshotManifest("yellow", "2023-01", "2023-02", files=[_record("a.parquet", "aa")])
    second = SnapshotManifest("yellow", "2023-01", "2023-02", files=[_record("a.parquet", "aa")])
    assert first.snapshot_id == second.snapshot_id


def test_snapshot_id_ignores_file_order() -> None:
    ordered = SnapshotManifest(
        "yellow",
        "2023-01",
        "2023-02",
        files=[_record("a.parquet", "aa"), _record("b.parquet", "bb")],
    )
    reversed_order = SnapshotManifest(
        "yellow",
        "2023-01",
        "2023-02",
        files=[_record("b.parquet", "bb"), _record("a.parquet", "aa")],
    )
    assert ordered.snapshot_id == reversed_order.snapshot_id


def test_snapshot_id_changes_when_upstream_file_is_revised() -> None:
    """TLC revises published files; a revision must produce a different ID."""
    original = SnapshotManifest("yellow", "2023-01", "2023-02", files=[_record("a.parquet", "aa")])
    revised = SnapshotManifest("yellow", "2023-01", "2023-02", files=[_record("a.parquet", "zz")])
    assert original.snapshot_id != revised.snapshot_id


def test_snapshot_id_changes_with_window() -> None:
    a = SnapshotManifest("yellow", "2023-01", "2023-02", files=[_record("a.parquet", "aa")])
    b = SnapshotManifest("yellow", "2023-01", "2023-03", files=[_record("a.parquet", "aa")])
    assert a.snapshot_id != b.snapshot_id


def test_total_rows() -> None:
    manifest = SnapshotManifest(
        "yellow",
        "2023-01",
        "2023-02",
        files=[_record("a.parquet", "aa"), _record("b.parquet", "bb")],
    )
    assert manifest.total_rows == 2000


def test_schema_consistency_detected() -> None:
    consistent = SnapshotManifest(
        "yellow",
        "2023-01",
        "2023-02",
        files=[_record("a.parquet", "aa", "ffff"), _record("b.parquet", "bb", "ffff")],
    )
    assert consistent.schema_is_consistent()

    drifted = SnapshotManifest(
        "yellow",
        "2023-01",
        "2023-02",
        files=[_record("a.parquet", "aa", "ffff"), _record("b.parquet", "bb", "eeee")],
    )
    assert not drifted.schema_is_consistent()


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = SnapshotManifest(
        "yellow",
        "2023-01",
        "2023-02",
        files=[_record("a.parquet", "aa"), _record("b.parquet", "bb")],
    )
    path = manifest.write(tmp_path)
    assert path.exists()

    loaded = SnapshotManifest.read(path)
    assert loaded.snapshot_id == manifest.snapshot_id
    assert loaded.total_rows == manifest.total_rows
    assert len(loaded.files) == 2


def test_manifest_read_rejects_tampered_file(tmp_path: Path) -> None:
    """A hand-edited manifest must not pass silently."""
    import json

    manifest = SnapshotManifest("yellow", "2023-01", "2023-02", files=[_record("a.parquet", "aa")])
    path = manifest.write(tmp_path)

    data = json.loads(path.read_text())
    data["files"][0]["sha256"] = "tampered"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="inconsistent"):
        SnapshotManifest.read(path)
