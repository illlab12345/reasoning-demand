"""Shared loader utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    if isinstance(value, str):
        if not value.strip():
            return True
        if value.strip().lower() in {"nan", "nat", "<na>"}:
            return True
    return False


def first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """Return the first non-null value among candidate keys."""
    for k in keys:
        v = row.get(k)
        if not _is_missing(v):
            return v
    return default


def make_jsonable(value: Any) -> Any:
    """Recursively convert numpy/pandas/timestamp values to JSON-safe Python types."""
    if hasattr(value, "tolist") and callable(value.tolist) and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return make_jsonable(value.tolist())
        except (ValueError, TypeError):
            pass
    if isinstance(value, dict):
        return {str(k): make_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_jsonable(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return make_jsonable(value.item())
        except (ValueError, TypeError):
            pass
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@dataclass
class LoadedDataset:
    dataset: str
    records: list[dict[str, Any]]
    row_counts: dict[str, int] = field(default_factory=dict)
    raw_columns: dict[str, list[str]] = field(default_factory=dict)
    skipped_rows: dict[str, int] = field(default_factory=dict)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "records": len(self.records),
            "row_counts": self.row_counts,
            "raw_columns": self.raw_columns,
            "skipped_rows": self.skipped_rows,
        }
