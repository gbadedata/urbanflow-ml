"""Data quality reporting.

Aggregates per-month validation into a committed artefact. The point is not the
headline pass rate but the reason breakdown: a month where quarantine jumps, or
where a reason code appears that never appeared before, is a signal about the
source that should be visible without re-running the pipeline.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from urbanflow.validation.validate import MonthValidationResult


def check_conservation(results: list[MonthValidationResult]) -> list[str]:
    """Months where rows_out + quarantined does not equal rows_in.

    A row that is neither output nor attributed to a rule has been lost, which is
    a bug rather than a data-quality finding. Reported per month so the offending
    file is identifiable.
    """
    return [
        f"{r.month}: {r.rows_in:,} in but {r.rows_out:,} out + "
        f"{sum(r.reason_counts.values()):,} quarantined "
        f"({r.rows_in - r.rows_out - sum(r.reason_counts.values()):+,} unaccounted)"
        for r in results
        if r.rows_out + sum(r.reason_counts.values()) != r.rows_in
    ]


def build_report(
    results: list[MonthValidationResult], snapshot_id: str | None = None
) -> dict[str, object]:
    total_in = sum(r.rows_in for r in results)
    total_out = sum(r.rows_out for r in results)

    reason_totals: dict[str, int] = {}
    for result in results:
        for code, count in result.reason_counts.items():
            reason_totals[code] = reason_totals.get(code, 0) + count

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "snapshot_id": snapshot_id,
        "months": len(results),
        "rows_in": total_in,
        "rows_out": total_out,
        "rows_quarantined": total_in - total_out,
        "quarantine_rate": round((total_in - total_out) / total_in, 6) if total_in else 0.0,
        "reason_totals": dict(sorted(reason_totals.items(), key=lambda kv: -kv[1])),
        "conservation_ok": not check_conservation(results),
        "by_month": [r.to_dict() for r in results],
    }


def write_report(report: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    return path


def format_summary(report: dict[str, object]) -> str:
    """Human-readable summary for the terminal."""
    rate = report["quarantine_rate"]
    assert isinstance(rate, float)
    lines = [
        f"months:           {report['months']}",
        f"rows in:          {report['rows_in']:,}",
        f"rows validated:   {report['rows_out']:,}",
        f"rows quarantined: {report['rows_quarantined']:,} ({rate:.4%})",
        "",
        "reason breakdown:",
    ]
    reasons = report["reason_totals"]
    assert isinstance(reasons, dict)
    if not reasons:
        lines.append("  (none)")
    for code, count in reasons.items():
        lines.append(f"  {code:<22} {count:>12,}")
    return "\n".join(lines)
