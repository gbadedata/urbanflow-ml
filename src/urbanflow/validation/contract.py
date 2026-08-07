"""Data contract enforcement.

The ingest manifest compares whole-schema fingerprints, which is the right check
for "did anything change upstream" but the wrong one for "can we still read
this". TLC adds fee columns between releases and shipped identical zone IDs as
int64 then int32; neither can affect a pipeline that consumes four columns.

This module declares what those four columns must look like, checks a file
against that declaration by type *family*, and reports incompatibilities as
specific, actionable messages rather than a boolean.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from urbanflow.config import PROJECT_ROOT

DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "data_contracts" / "tlc_yellow_trips.yaml"

#: Arrow type prefixes belonging to each family. Compared by prefix so that
#: parameterised types (``timestamp[us]``, ``timestamp[ns, tz=UTC]``) match.
TYPE_FAMILIES: dict[str, tuple[str, ...]] = {
    "integer": ("int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"),
    "float": ("float", "double", "halffloat"),
    "timestamp": ("timestamp", "date32", "date64"),
    "string": ("string", "large_string", "utf8", "large_utf8"),
}


def type_family(arrow_type: str) -> str | None:
    """Map an Arrow type string to its family, or None if unrecognised."""
    lowered = arrow_type.lower()
    for family, prefixes in TYPE_FAMILIES.items():
        if lowered.startswith(prefixes):
            return family
    return None


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    type_family: str
    canonical_type: str
    role: str
    nullable: bool


@dataclass(frozen=True)
class RuleSpec:
    code: str
    description: str


@dataclass(frozen=True)
class DataContract:
    name: str
    version: int
    required_columns: tuple[ColumnSpec, ...]
    prohibited_columns: frozenset[str]
    rules: tuple[RuleSpec, ...]

    @property
    def required_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.required_columns)

    @property
    def rule_codes(self) -> frozenset[str]:
        return frozenset(r.code for r in self.rules)

    def check_schema(self, schema: dict[str, str]) -> list[str]:
        """Return a list of violations; empty means the file is readable.

        ``schema`` maps column name to Arrow type string.
        """
        problems: list[str] = []
        for spec in self.required_columns:
            if spec.name not in schema:
                problems.append(f"missing required column {spec.name!r} ({spec.role})")
                continue
            actual = schema[spec.name]
            family = type_family(actual)
            if family is None:
                problems.append(f"column {spec.name!r} has unrecognised type {actual!r}")
            elif family != spec.type_family:
                problems.append(
                    f"column {spec.name!r} is {actual!r} (family {family!r}) but the "
                    f"contract requires family {spec.type_family!r}"
                )
        return problems

    def prohibited_present(self, schema: dict[str, str]) -> list[str]:
        """Prohibited columns found in the source.

        Their presence in raw data is expected and harmless — the contract exists
        so that a test can assert they never reach a model (spec §7.4).
        """
        return sorted(set(schema) & self.prohibited_columns)


def load_contract(path: Path | None = None) -> DataContract:
    with (path or DEFAULT_CONTRACT_PATH).open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    columns = tuple(
        ColumnSpec(
            name=c["name"],
            type_family=c["type_family"],
            canonical_type=c["canonical_type"],
            role=c["role"],
            nullable=c["nullable"],
        )
        for c in raw["required_columns"]
    )
    rules = tuple(RuleSpec(code=r["code"], description=r["description"]) for r in raw["rules"])

    unknown = {c.type_family for c in columns} - set(TYPE_FAMILIES)
    if unknown:
        raise ValueError(f"contract references unknown type families: {sorted(unknown)}")

    return DataContract(
        name=raw["name"],
        version=raw["version"],
        required_columns=columns,
        prohibited_columns=frozenset(raw["prohibited_columns"]),
        rules=rules,
    )


def parquet_schema(path: Path) -> dict[str, str]:
    """Column name to Arrow type string, read from footer metadata only."""
    import pyarrow.parquet as pq

    schema = pq.ParquetFile(path).schema_arrow
    return {field.name: str(field.type) for field in schema}
