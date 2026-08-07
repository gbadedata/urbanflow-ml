"""Command-line entrypoint.

uv run urbanflow config-show
uv run urbanflow ingest --zones
uv run urbanflow manifest-show <snapshot_id>
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from urbanflow.config import get_config
from urbanflow.ingest.download import build_snapshot, download_zone_lookup
from urbanflow.ingest.manifest import SnapshotManifest
from urbanflow.ingest.sources import source_files

app = typer.Typer(add_completion=False, help="UrbanFlow ML pipeline commands.")


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )


@app.command("config-show")
def config_show() -> None:
    """Print the resolved configuration and the files it implies."""
    config = get_config()
    files = source_files(config.data.service, config.data.start_month, config.data.end_month)
    typer.echo(f"service:          {config.data.service}")
    typer.echo(f"window:           {config.data.start_month} .. {config.data.end_month}")
    typer.echo(f"months:           {len(files)}")
    typer.echo(f"raw dir:          {config.raw_path}")
    typer.echo(f"manifest dir:     {config.manifest_path}")
    typer.echo(f"excluded zones:   {config.zones.excluded_location_ids}")
    typer.echo(f"feature cutoff:   {config.features.feature_cutoff_minutes} min")
    typer.echo(f"horizons:         {config.forecast.horizons_hours} h")
    typer.echo(f"holdout:          {config.split.holdout_weeks} weeks")


@app.command("ingest")
def ingest(
    force: bool = typer.Option(False, "--force", help="Re-download files already present."),
    zones: bool = typer.Option(True, "--zones/--no-zones", help="Also fetch the zone lookup."),
) -> None:
    """Download the configured window and write a snapshot manifest."""
    config = get_config()
    if zones:
        path = download_zone_lookup(config, force=force)
        typer.echo(f"zone lookup: {path}")

    manifest = build_snapshot(config, force=force)
    written = manifest.write(config.manifest_path)

    typer.echo("")
    typer.echo(f"snapshot_id:  {manifest.snapshot_id}")
    typer.echo(f"files:        {len(manifest.files)}")
    typer.echo(f"total rows:   {manifest.total_rows:,}")
    typer.echo(f"schema consistent: {manifest.schema_is_consistent()}")
    typer.echo(f"manifest:     {written}")


@app.command("manifest-show")
def manifest_show(snapshot_id: str) -> None:
    """Print a stored manifest summary."""
    config = get_config()
    path: Path = config.manifest_path / f"{snapshot_id}.json"
    if not path.exists():
        typer.echo(f"no manifest at {path}", err=True)
        raise typer.Exit(code=1)
    manifest = SnapshotManifest.read(path)
    typer.echo(f"snapshot_id:  {manifest.snapshot_id}")
    typer.echo(f"window:       {manifest.start_month} .. {manifest.end_month}")
    typer.echo(f"created:      {manifest.created_at}")
    typer.echo(f"total rows:   {manifest.total_rows:,}")
    for record in manifest.files:
        typer.echo(f"  {record.filename}  {record.rows:>12,} rows  {record.sha256[:12]}")


if __name__ == "__main__":
    app()
