"""Source snapshot manifest.

Spec v2 §6.1: every experiment references a snapshot ID. The manifest records
what was retrieved, when, how big it was, how many rows it held and what shape
those rows had — so a result can be traced back to the exact bytes that produced
it even though TLC republishes and revises files over time.

The snapshot ID is derived from the file checksums themselves, so two people who
download the same window get the same ID, and a silently revised upstream file
produces a different one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class FileRecord:
    """Provenance for a single retrieved file."""

    filename: str
    url: str
    sha256: str
    bytes: int
    rows: int
    schema_fingerprint: str
    retrieved_at: str


@dataclass
class SnapshotManifest:
    """Provenance for a full retrieval window."""

    service: str
    start_month: str
    end_month: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    files: list[FileRecord] = field(default_factory=list)

    @property
    def snapshot_id(self) -> str:
        """Deterministic ID derived from the content of every file in the window.

        Sorted so that retrieval order cannot change the ID.
        """
        digest = hashlib.sha256()
        digest.update(f"{self.service}|{self.start_month}|{self.end_month}".encode())
        for record in sorted(self.files, key=lambda r: r.filename):
            digest.update(record.sha256.encode())
        return f"{self.service}-{self.start_month}-{self.end_month}-{digest.hexdigest()[:12]}"

    @property
    def total_rows(self) -> int:
        return sum(r.rows for r in self.files)

    def schema_is_consistent(self) -> bool:
        """True when every file in the window shares one schema.

        Spec §6.1 treats a schema change as an explicit migration. A False here
        means the window straddles a TLC schema revision and must be handled
        deliberately rather than cast silently.
        """
        fingerprints = {r.schema_fingerprint for r in self.files}
        return len(fingerprints) <= 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "service": self.service,
            "start_month": self.start_month,
            "end_month": self.end_month,
            "created_at": self.created_at,
            "file_count": len(self.files),
            "total_rows": self.total_rows,
            "schema_consistent": self.schema_is_consistent(),
            "files": [asdict(r) for r in self.files],
        }

    def write(self, manifest_dir: Path) -> Path:
        manifest_dir.mkdir(parents=True, exist_ok=True)
        path = manifest_dir / f"{self.snapshot_id}.json"
        with path.open("w") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        return path

    @classmethod
    def read(cls, path: Path) -> SnapshotManifest:
        with path.open() as fh:
            data = json.load(fh)
        manifest = cls(
            service=data["service"],
            start_month=data["start_month"],
            end_month=data["end_month"],
            created_at=data["created_at"],
            files=[FileRecord(**r) for r in data["files"]],
        )
        if manifest.snapshot_id != data["snapshot_id"]:
            raise ValueError(
                f"manifest {path} is inconsistent: stored snapshot_id "
                f"{data['snapshot_id']!r} does not match recomputed "
                f"{manifest.snapshot_id!r}"
            )
        return manifest


def sha256_file(path: Path) -> str:
    """Streaming checksum, so large Parquet files never land in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def schema_fingerprint(names_and_types: list[tuple[str, str]]) -> str:
    """Stable hash of a column schema.

    Sorted by column name so that column reordering upstream — which is harmless —
    does not read as a schema change, while an added, removed or retyped column does.
    """
    canonical = ";".join(f"{name}:{dtype}" for name, dtype in sorted(names_and_types))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def parquet_metadata(path: Path) -> tuple[int, str]:
    """Row count and schema fingerprint, read from Parquet footer metadata only.

    Deliberately does not load the data: a monthly TLC file is millions of rows and
    the footer already carries everything needed here.
    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    rows = parquet_file.metadata.num_rows
    arrow_schema = parquet_file.schema_arrow
    pairs = [(field.name, str(field.type)) for field in arrow_schema]
    return rows, schema_fingerprint(pairs)
