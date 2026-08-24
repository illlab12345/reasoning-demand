"""Unified internal record schema (v0.1) and validation."""

from __future__ import annotations

import math
from typing import Any, Iterable

SCHEMA_VERSION = "v0.1"

REQUIRED_FIELDS = (
    "id",
    "source_id",
    "dataset",
    "domain",
    "question",
    "answer",
    "difficulty",
    "metadata",
)


def validate_record(record: Any, dataset: str | None = None) -> list[str]:
    """Return a list of validation errors (empty means valid)."""
    errors: list[str] = []
    if not isinstance(record, dict):
        return ["record is not a dict"]

    for field in REQUIRED_FIELDS:
        if field not in record:
            errors.append(f"missing field: {field}")
    if errors:
        return errors

    if not isinstance(record["id"], str) or not record["id"].strip():
        errors.append("id must be a non-empty string")
    if not isinstance(record["source_id"], str) or not record["source_id"].strip():
        errors.append("source_id must be a non-empty string")
    if not isinstance(record["dataset"], str) or not record["dataset"].strip():
        errors.append("dataset must be a non-empty string")
    if not isinstance(record["domain"], str) or not record["domain"].strip():
        errors.append("domain must be a non-empty string")
    if not isinstance(record["question"], str) or not record["question"].strip():
        errors.append("question must be a non-empty string")
    if not isinstance(record["answer"], str) or not record["answer"].strip():
        errors.append("answer must be a non-empty string")

    if dataset is not None and record["dataset"] != dataset:
        errors.append(f"dataset mismatch: {record['dataset']!r} != {dataset!r}")

    diff = record["difficulty"]
    if diff is not None:
        if isinstance(diff, bool) or not isinstance(diff, (int, float)):
            errors.append("difficulty must be null or a number")
        elif isinstance(diff, float) and math.isnan(diff):
            errors.append("difficulty must not be NaN")

    if not isinstance(record["metadata"], dict):
        errors.append("metadata must be a dict")

    return errors


def validate_records(records: Iterable[dict[str, Any]], dataset: str | None = None) -> dict[str, Any]:
    """Validate a full dataset; returns summary with errors."""
    ids: list[str] = []
    errors: list[str] = []
    for rec in records:
        ids.append(rec.get("id", "<missing>"))
        errors.extend(validate_record(rec, dataset=dataset))

    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    return {
        "n": len(ids),
        "n_errors": len(errors),
        "errors": errors[:100],
        "unique_ids": len(set(ids)) == len(ids),
        "duplicate_ids": duplicates[:20],
    }

