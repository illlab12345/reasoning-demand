"""Dataset statistics for processed (unified-schema) data."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _difficulty_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    vals = [r["difficulty"] for r in records if r.get("difficulty") is not None]
    if not vals:
        return {"present": 0, "min": None, "max": None, "mean": None}
    return {
        "present": len(vals),
        "min": min(vals),
        "max": max(vals),
        "mean": round(sum(vals) / len(vals), 4),
    }


def compute_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [r["id"] for r in records]
    missing: dict[str, int] = {}
    for field in ("question", "answer", "source_id", "dataset", "domain"):
        missing[field] = sum(1 for r in records if not r.get(field))
    ans_lens = [len(r["answer"]) for r in records]
    return {
        "n": len(records),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "missing_fields": missing,
        "domain_counts": dict(Counter(r["domain"] for r in records)),
        "metadata_keys": sorted({k for r in records for k in r.get("metadata", {})}),
        "difficulty": _difficulty_stats(records),
        "answer_length": {
            "min": min(ans_lens),
            "max": max(ans_lens),
            "mean": round(sum(ans_lens) / len(ans_lens), 2),
        },
    }


def format_statistics(stats: dict[str, Any]) -> str:
    lines = [
        f"rows: {stats['n']}",
        f"unique ids: {stats['unique_ids']} (duplicates: {stats['duplicate_ids']})",
        f"missing fields: {stats['missing_fields']}",
        f"domains: {stats['domain_counts']}",
        f"difficulty: {stats['difficulty']}",
        f"answer length: {stats['answer_length']}",
        f"metadata keys: {stats['metadata_keys']}",
    ]
    return "\n".join(lines)
