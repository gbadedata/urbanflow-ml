"""Command-line entrypoint.

uv run urbanflow config-show
uv run urbanflow ingest --zones
uv run urbanflow manifest-show <snapshot_id>
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import typer

from urbanflow.config import PROJECT_ROOT, get_config
from urbanflow.duck import connect
from urbanflow.features.build import (
    assert_no_future_leakage,
    build_features,
    feature_columns,
)
from urbanflow.features.grid import (
    build_grid,
    classify_zero_hour,
    detect_city_wide_zero_hours,
)
from urbanflow.ingest.download import build_snapshot, download_zone_lookup
from urbanflow.ingest.manifest import SnapshotManifest
from urbanflow.ingest.sources import month_range, source_files
from urbanflow.models.baselines import BASELINE_B
from urbanflow.models.champion import CHAMPION, evaluate_gate, format_gate
from urbanflow.models.evaluate import (
    evaluate_baselines,
    format_results,
    load_zone_tiers,
)
from urbanflow.models.intervals import (
    check_coverage_gate,
    estimate_dispersion,
    format_coverage,
    interval_bounds,
    measure_coverage_by_tier,
)
from urbanflow.models.split import (
    CALIBRATE,
    TEST,
    assert_no_temporal_overlap,
    assign_blocks,
    block_counts,
    compute_bounds,
)
from urbanflow.validation.contract import load_contract, parquet_schema
from urbanflow.validation.quality import (
    build_report,
    check_conservation,
    format_summary,
    write_report,
)
from urbanflow.validation.validate import validate_month
from urbanflow.validation.zones import (
    forecastable_zones,
    profile_zones,
    write_zone_registry,
)

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


@app.command("contract-check")
def contract_check() -> None:
    """Check every raw file against the data contract."""
    config = get_config()
    contract = load_contract()
    months = month_range(config.data.start_month, config.data.end_month)

    failures = 0
    for month in months:
        path = config.raw_path / f"{config.data.service}_tripdata_{month}.parquet"
        if not path.exists():
            typer.echo(f"{month}  MISSING")
            failures += 1
            continue
        problems = contract.check_schema(parquet_schema(path))
        if problems:
            failures += 1
            typer.echo(f"{month}  FAIL")
            for problem in problems:
                typer.echo(f"          {problem}")

    if failures:
        typer.echo(f"\n{failures} of {len(months)} files fail the contract", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"all {len(months)} files satisfy contract {contract.name} v{contract.version}")


@app.command("validate")
def validate(
    dry_run: bool = typer.Option(False, "--dry-run", help="Count only; write nothing."),
) -> None:
    """Validate every month, quarantine failures and write the quality report."""
    config = get_config()
    contract = load_contract()
    months = month_range(config.data.start_month, config.data.end_month)

    con = connect()
    results = []
    for month in months:
        path = config.raw_path / f"{config.data.service}_tripdata_{month}.parquet"
        problems = contract.check_schema(parquet_schema(path))
        if problems:
            typer.echo(f"{month}: contract violation, refusing to validate", err=True)
            for problem in problems:
                typer.echo(f"  {problem}", err=True)
            raise typer.Exit(code=1)

        result = validate_month(con, config, month, write=not dry_run)
        results.append(result)
        typer.echo(
            f"{month}  in {result.rows_in:>10,}  out {result.rows_out:>10,}  "
            f"quarantined {result.rows_quarantined:>8,} ({result.quarantine_rate:.3%})"
        )

    report = build_report(results)
    typer.echo("")
    typer.echo(format_summary(report))

    lost = check_conservation(results)
    if lost:
        typer.echo("\nROWS UNACCOUNTED FOR — this is a bug, not a data finding:", err=True)
        for line in lost:
            typer.echo(f"  {line}", err=True)
        raise typer.Exit(code=1)

    if not dry_run:
        path = write_report(report, PROJECT_ROOT / "reports" / "data_quality" / "validation.json")
        typer.echo(f"\nreport: {path}")


@app.command("zone-report")
def zone_report() -> None:
    """Profile zone volumes and write the forecastable-zone registry."""
    config = get_config()
    con = connect()

    profiles = profile_zones(con, config)
    if not profiles:
        typer.echo("no validated data — run `urbanflow validate` first", err=True)
        raise typer.Exit(code=1)

    keep = forecastable_zones(profiles, config)
    path = write_zone_registry(profiles, config, PROJECT_ROOT / "reports" / "zone_registry.json")

    total = sum(p.trips for p in profiles)
    kept_trips = sum(p.trips for p in profiles if p.zone_id in set(keep))
    typer.echo(f"zones profiled:     {len(profiles)}")
    typer.echo(f"zones forecastable: {len(keep)}")
    typer.echo(f"trips retained:     {kept_trips / total:.4%}" if total else "no trips")
    typer.echo(f"registry:           {path}")


@app.command("build-grid")
def build_grid_cmd() -> None:
    """Build the dense zone-hour grid and assert it is complete."""
    import json

    config = get_config()
    registry_path = PROJECT_ROOT / "reports" / "zone_registry.json"
    if not registry_path.exists():
        typer.echo("no zone registry — run `urbanflow zone-report` first", err=True)
        raise typer.Exit(code=1)

    registry = json.loads(registry_path.read_text())
    zone_ids = [z["zone_id"] for z in registry["zones"] if z["tier"] != "excluded"]

    con = connect()
    stats = build_grid(con, config, zone_ids)

    typer.echo(f"zones:          {stats.zones}")
    typer.echo(f"hours:          {stats.hours:,}")
    typer.echo(f"rows:           {stats.rows:,}")
    typer.echo(f"expected rows:  {stats.expected_rows:,}")
    typer.echo(f"zero rows:      {stats.zero_rows:,} ({stats.zero_share:.2%})")
    typer.echo(f"total pickups:  {stats.total_pickups:,}")
    typer.echo(f"span:           {stats.start_hour} .. {stats.end_hour}")

    gaps = detect_city_wide_zero_hours(con)
    if gaps:
        classified = [(g, classify_zero_hour(g)) for g in gaps]
        artifacts = [g for g, kind in classified if kind == "dst_fold_artifact"]
        real_gaps = [g for g, kind in classified if kind == "ingestion_gap"]

        typer.echo(f"\ncity-wide zero hours: {len(gaps)} (excluded from training)")
        typer.echo(f"  DST fold artifacts: {len(artifacts)} (expected, ~1/year)")
        typer.echo(f"  ingestion gaps:     {len(real_gaps)}")
        for gap in real_gaps[:10]:
            typer.echo(f"    {gap}")

        excluded = config.validated_path.parent / "grid" / "excluded_hours.json"
        excluded.parent.mkdir(parents=True, exist_ok=True)
        excluded.write_text(
            json.dumps(
                [{"hour_utc": g.isoformat(), "reason": kind} for g, kind in classified],
                indent=2,
            )
        )
        typer.echo(f"  written: {excluded}")

    out = config.validated_path.parent / "grid" / "zone_hour.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY zone_hour TO '{out}' (FORMAT PARQUET)")
    typer.echo(f"\ngrid: {out}")

    stats_path = PROJECT_ROOT / "reports" / "grid_stats.json"
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2))
    typer.echo(f"stats: {stats_path}")

    if not stats.is_dense:
        typer.echo(
            f"\nGRID IS NOT DENSE: {stats.rows:,} rows but {stats.expected_rows:,} "
            "expected. This is a release blocker (spec §8, gate 1).",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo("\ndensity check: PASS")


@app.command("build-features")
def build_features_cmd() -> None:
    """Build one feature table per horizon and assert no future leakage."""
    import json

    config = get_config()
    grid_path = config.validated_path.parent / "grid" / "zone_hour.parquet"
    if not grid_path.exists():
        typer.echo("no grid — run `urbanflow build-grid` first", err=True)
        raise typer.Exit(code=1)

    con = connect()
    con.execute(
        f"CREATE OR REPLACE TEMP VIEW zone_hour AS SELECT * FROM read_parquet('{grid_path}')"
    )

    out_dir = config.validated_path.parent / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    for horizon in config.forecast.horizons_hours:
        table = build_features(con, config, horizon)
        assert_no_future_leakage(con, config, table, horizon)

        row = con.execute(f"SELECT COUNT(*), MIN(target_hour), MAX(target_hour) FROM {table}")
        count, first, last = row.fetchone()  # type: ignore[misc]
        cols = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {table}").fetchall()]
        features = feature_columns(cols)

        out = out_dir / f"{table}.parquet"
        con.execute(f"COPY {table} TO '{out}' (FORMAT PARQUET)")

        typer.echo(
            f"h={horizon}h  rows {count:>10,}  features {len(features):>2}  "
            f"anchor = target - {horizon + 2}h  leakage check PASS"
        )
        summary.append(
            {
                "horizon_hours": horizon,
                "rows": count,
                "anchor_offset_hours": horizon + 2,
                "feature_count": len(features),
                "features": features,
                "first_target": str(first),
                "last_target": str(last),
                "path": str(out),
            }
        )

    path = PROJECT_ROOT / "reports" / "feature_summary.json"
    path.write_text(json.dumps(summary, indent=2))
    typer.echo(f"\nsummary: {path}")


@app.command("baselines")
def baselines_cmd(
    frozen_test: bool = typer.Option(
        False,
        "--frozen-test",
        help="Read the FROZEN test block. Spec §6.2: once, at release.",
    ),
    champion: bool = typer.Option(
        False, "--champion", help="Also fit the GBM champion and apply the promotion gate."
    ),
    max_iter: int = typer.Option(
        400,
        "--max-iter",
        help="Champion boosting iterations. Raise if early stopping never fires.",
    ),
) -> None:
    """Fit on train and evaluate the ladder on the calibrate block."""
    import json

    config = get_config()
    feature_dir = config.validated_path.parent / "features"
    registry = PROJECT_ROOT / "reports" / "zone_registry.json"
    if not registry.exists():
        typer.echo("no zone registry — run `urbanflow zone-report` first", err=True)
        raise typer.Exit(code=1)

    eval_block = TEST if frozen_test else CALIBRATE
    if frozen_test:
        typer.echo("READING THE FROZEN TEST BLOCK. This should happen once, at release.")

    zone_tiers = load_zone_tiers(registry)
    summary = json.loads((PROJECT_ROOT / "reports" / "feature_summary.json").read_text())
    con = connect()
    report: dict[str, object] = {"eval_block": eval_block, "horizons": {}}

    for entry in summary:
        horizon = entry["horizon_hours"]
        path = feature_dir / f"features_h{horizon}.parquet"
        if not path.exists():
            typer.echo(f"missing {path} — run `urbanflow build-features`", err=True)
            raise typer.Exit(code=1)

        con.execute(f"CREATE OR REPLACE TEMP TABLE f AS SELECT * FROM read_parquet('{path}')")
        span = con.execute("SELECT MIN(target_hour), MAX(target_hour) FROM f").fetchone()
        assert span is not None
        bounds = compute_bounds(span[0], span[1], config)
        assign_blocks(con, "f", bounds)
        assert_no_temporal_overlap(con, "f")

        counts = block_counts(con, "f")
        typer.echo(
            f"\nh={horizon}h blocks: " + "  ".join(f"{k}={v:,}" for k, v in sorted(counts.items()))
        )

        results = evaluate_baselines(
            con,
            config,
            "f",
            entry["features"],
            zone_tiers,
            eval_block,
            with_champion=champion,
            max_iter=max_iter,
        )
        typer.echo(format_results(results, horizon))

        gate = None
        if champion:
            if CHAMPION not in results:
                typer.echo(f"champion requested but {CHAMPION!r} is absent from results", err=True)
                raise typer.Exit(code=1)
            gate = evaluate_gate(config, horizon, results[CHAMPION], results[BASELINE_B])
            typer.echo(format_gate(gate))

        report["horizons"][str(horizon)] = {  # type: ignore[index]
            "bounds": bounds.to_dict(),
            "block_counts": counts,
            "results": {name: [m.to_dict() for m in metrics] for name, metrics in results.items()},
            "gate": gate.to_dict() if gate else None,
        }

    stem = "champion" if champion else "baselines"
    name = f"{stem}_test.json" if frozen_test else f"{stem}_calibrate.json"
    path = PROJECT_ROOT / "reports" / name
    path.write_text(json.dumps(report, indent=2))
    typer.echo(f"\nreport: {path}")


@app.command("intervals")
def intervals_cmd(
    max_iter: int = typer.Option(400, "--max-iter", help="Champion boosting iterations."),
    frozen_test: bool = typer.Option(
        False, "--frozen-test", help="Read the FROZEN test block. Spec §6.2: once, at release."
    ),
) -> None:
    """Fit dispersion on calibrate, then measure interval coverage.

    During development the calibrate block is split in half by time: dispersion
    is fitted on the earlier half and coverage verified on the later one, so the
    reported figure is never in-sample for the parameter that produced it. At
    release, dispersion is fitted on the whole calibrate block and coverage is
    measured once on the frozen test block.
    """
    import json

    from urbanflow.models.champion import fit_champion
    from urbanflow.models.evaluate import load_block

    config = get_config()
    feature_dir = config.validated_path.parent / "features"
    registry = PROJECT_ROOT / "reports" / "zone_registry.json"
    if not registry.exists():
        typer.echo("no zone registry — run `urbanflow zone-report` first", err=True)
        raise typer.Exit(code=1)

    zone_tiers = load_zone_tiers(registry)
    summary = json.loads((PROJECT_ROOT / "reports" / "feature_summary.json").read_text())
    con = connect()
    report: dict[str, object] = {"frozen_test": frozen_test, "horizons": {}}
    all_passed = True

    for entry in summary:
        horizon = entry["horizon_hours"]
        path = feature_dir / f"features_h{horizon}.parquet"
        con.execute(f"CREATE OR REPLACE TEMP TABLE f AS SELECT * FROM read_parquet('{path}')")
        span = con.execute("SELECT MIN(target_hour), MAX(target_hour) FROM f").fetchone()
        assert span is not None
        assign_blocks(con, "f", compute_bounds(span[0], span[1], config))

        cols = entry["features"]
        train_x, train_y, _ = load_block(con, "f", "train", cols)
        typer.echo(f"\nh={horizon}h fitting champion on {len(train_y):,} rows")
        model = fit_champion(train_x, train_y, cols, max_iter=max_iter)

        cal_x, cal_y, cal_zones = load_block(con, "f", CALIBRATE, cols)
        cal_mu = model.predict(cal_x)

        if frozen_test:
            typer.echo("READING THE FROZEN TEST BLOCK. This should happen once, at release.")
            dispersion = estimate_dispersion(cal_y, cal_mu)
            eval_x, eval_y, eval_zones = load_block(con, "f", TEST, cols)
            eval_mu = model.predict(eval_x)
        else:
            half = len(cal_y) // 2
            dispersion = estimate_dispersion(cal_y[:half], cal_mu[:half])
            eval_y, eval_zones, eval_mu = cal_y[half:], cal_zones[half:], cal_mu[half:]

        lower, upper = interval_bounds(eval_mu, dispersion)
        tiers = np.array([zone_tiers.get(int(z), "low") for z in eval_zones])
        results = measure_coverage_by_tier(eval_y, lower, upper, tiers)
        typer.echo(format_coverage(results, horizon, dispersion))

        passed, reasons = check_coverage_gate(
            results, config.gates.gated_tiers, config.gates.coverage_tolerance
        )
        all_passed = all_passed and passed
        typer.echo(f"\n  coverage gate: {'PASS' if passed else 'FAIL'}")
        for reason in reasons:
            typer.echo(f"  - {reason}")

        report["horizons"][str(horizon)] = {  # type: ignore[index]
            "dispersion": round(dispersion, 4),
            "coverage": [r.to_dict() for r in results],
            "gate_passed": passed,
            "reasons": reasons,
        }

    name = "intervals_test.json" if frozen_test else "intervals_calibrate.json"
    out = PROJECT_ROOT / "reports" / name
    out.write_text(json.dumps(report, indent=2))
    typer.echo(f"\nreport: {out}")
    if not all_passed:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
