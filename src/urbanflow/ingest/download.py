"""Retrieval of TLC monthly Parquet files.

Downloads are idempotent: a file already present with a readable Parquet footer
is not re-fetched. Partial downloads are written to a temporary path and moved
into place only on success, so an interrupted run never leaves a truncated file
that looks complete on the next pass.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from urbanflow.config import Config
from urbanflow.ingest.manifest import (
    FileRecord,
    SnapshotManifest,
    parquet_metadata,
    sha256_file,
)
from urbanflow.ingest.sources import SourceFile, source_files

logger = logging.getLogger(__name__)

TIMEOUT = httpx.Timeout(30.0, read=300.0)
MAX_ATTEMPTS = 3


def _is_valid_parquet(path: Path) -> bool:
    """Cheap integrity check: can the footer be read at all?"""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        parquet_metadata(path)
    except Exception:
        return False
    return True


def download_file(source: SourceFile, dest_dir: Path, *, force: bool = False) -> Path:
    """Fetch one monthly file, returning its local path.

    Skips the download when a valid file is already present unless ``force``.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / source.filename

    if not force and _is_valid_parquet(dest):
        logger.info("skip %s (already present)", source.filename)
        return dest

    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            logger.info("downloading %s (attempt %d)", source.filename, attempt)
            with tempfile.NamedTemporaryFile(dir=dest_dir, delete=False, suffix=".part") as tmp:
                tmp_path = Path(tmp.name)
                with httpx.stream("GET", source.url, timeout=TIMEOUT, follow_redirects=True) as r:
                    r.raise_for_status()
                    for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                        tmp.write(chunk)
            if not _is_valid_parquet(tmp_path):
                tmp_path.unlink(missing_ok=True)
                raise ValueError(f"{source.filename} downloaded but is not readable Parquet")
            shutil.move(str(tmp_path), dest)
            return dest
        except Exception as exc:  # noqa: BLE001 - retried and re-raised below
            last_error = exc
            logger.warning("attempt %d failed for %s: %s", attempt, source.filename, exc)

    raise RuntimeError(
        f"failed to download {source.filename} after {MAX_ATTEMPTS} attempts: {last_error}"
    )


def build_snapshot(config: Config, *, force: bool = False) -> SnapshotManifest:
    """Download the configured window and return its manifest.

    The manifest is the unit of reproducibility; see spec §6.1.
    """
    sources = source_files(config.data.service, config.data.start_month, config.data.end_month)
    manifest = SnapshotManifest(
        service=config.data.service,
        start_month=config.data.start_month,
        end_month=config.data.end_month,
    )

    for source in sources:
        path = download_file(source, config.raw_path, force=force)
        rows, fingerprint = parquet_metadata(path)
        manifest.files.append(
            FileRecord(
                filename=source.filename,
                url=source.url,
                sha256=sha256_file(path),
                bytes=path.stat().st_size,
                rows=rows,
                schema_fingerprint=fingerprint,
                retrieved_at=datetime.now(UTC).isoformat(),
            )
        )
        logger.info("%s: %s rows", source.filename, f"{rows:,}")

    if not manifest.schema_is_consistent():
        logger.warning(
            "schema differs across files in this window — treat as an explicit "
            "migration (spec §6.1), do not cast silently"
        )

    return manifest


def download_zone_lookup(config: Config, *, force: bool = False) -> Path:
    """Fetch the TLC taxi zone lookup CSV.

    Needed before the dense grid can be built, since it defines which
    LocationIDs exist at all (spec §2.2).
    """
    dest = config.raw_path / "taxi_zone_lookup.csv"
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(config.zones.lookup_url, timeout=TIMEOUT, follow_redirects=True)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest
