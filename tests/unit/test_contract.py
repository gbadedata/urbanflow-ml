"""Tests for the data contract.

The contract exists because the ingest manifest's whole-schema fingerprint is too
blunt: it reported the real 30-file snapshot as inconsistent because TLC added a
congestion-fee column and changed an integer width. These tests pin the
distinction between changes that matter and changes that do not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from urbanflow.validation.contract import (
    load_contract,
    parquet_schema,
    type_family,
)


def test_contract_loads() -> None:
    contract = load_contract()
    assert contract.name == "tlc_yellow_trips"
    assert "tpep_pickup_datetime" in contract.required_names
    assert "PULocationID" in contract.required_names


def test_type_family_groups_integer_widths() -> None:
    """The 2023-01 -> 2023-02 zone retype (int64 -> int32) must be benign."""
    assert type_family("int64") == type_family("int32") == "integer"


def test_type_family_groups_string_variants() -> None:
    assert type_family("string") == type_family("large_string") == "string"


def test_type_family_distinguishes_incompatible_types() -> None:
    assert type_family("int64") != type_family("double")
    assert type_family("timestamp[us]") != type_family("string")


def test_type_family_handles_parameterised_timestamps() -> None:
    assert type_family("timestamp[us]") == "timestamp"
    assert type_family("timestamp[ns, tz=UTC]") == "timestamp"


def test_type_family_unknown_returns_none() -> None:
    assert type_family("decimal128(5, 2)") is None


def test_modern_schema_passes(clean_month: Path) -> None:
    contract = load_contract()
    assert contract.check_schema(parquet_schema(clean_month)) == []


def test_legacy_2023_01_schema_passes(legacy_month: Path) -> None:
    """The real snapshot's odd month out must still satisfy the contract."""
    contract = load_contract()
    assert contract.check_schema(parquet_schema(legacy_month)) == []


def test_added_column_does_not_break_contract(clean_month: Path) -> None:
    """TLC added cbd_congestion_fee in 2025-01; we do not consume it."""
    contract = load_contract()
    schema = parquet_schema(clean_month)
    schema["cbd_congestion_fee"] = "double"
    assert contract.check_schema(schema) == []


def test_missing_required_column_fails() -> None:
    contract = load_contract()
    problems = contract.check_schema({"PULocationID": "int32"})
    assert any("tpep_pickup_datetime" in p for p in problems)


def test_incompatible_type_fails() -> None:
    """A zone ID shipped as text is a real migration, not a widening."""
    contract = load_contract()
    schema = {
        "tpep_pickup_datetime": "timestamp[us]",
        "tpep_dropoff_datetime": "timestamp[us]",
        "PULocationID": "string",
        "VendorID": "int32",
    }
    problems = contract.check_schema(schema)
    assert len(problems) == 1
    assert "PULocationID" in problems[0]
    assert "integer" in problems[0]


def test_prohibited_columns_detected_in_source(clean_month: Path) -> None:
    """Fare fields exist in raw data; the contract's job is to name them."""
    contract = load_contract()
    found = contract.prohibited_present(parquet_schema(clean_month))
    assert "fare_amount" in found
    assert "total_amount" in found
    assert "DOLocationID" in found


def test_required_and_prohibited_do_not_overlap() -> None:
    """A column cannot be both consumed and forbidden."""
    contract = load_contract()
    assert not set(contract.required_names) & contract.prohibited_columns


def test_dropoff_is_declared_qc_only() -> None:
    """Spec §7.4: dropoff time must never become a feature."""
    contract = load_contract()
    spec = next(c for c in contract.required_columns if c.name == "tpep_dropoff_datetime")
    assert "quality control" in spec.role.lower()


def test_all_rule_codes_unique() -> None:
    contract = load_contract()
    codes = [r.code for r in contract.rules]
    assert len(codes) == len(set(codes))


def test_contract_rejects_unknown_type_family(tmp_path: Path) -> None:
    import yaml

    from urbanflow.validation.contract import DEFAULT_CONTRACT_PATH

    raw = yaml.safe_load(DEFAULT_CONTRACT_PATH.read_text())
    raw["required_columns"][0]["type_family"] = "quaternion"
    path = tmp_path / "contract.yaml"
    path.write_text(yaml.safe_dump(raw))

    with pytest.raises(ValueError, match="unknown type families"):
        load_contract(path)
