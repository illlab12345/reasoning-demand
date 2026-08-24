"""MATH-500 loader (HuggingFaceH4/MATH-500, pinned revision)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..io import read_jsonl
from .base import LoadedDataset, first_present


def load_math500(raw_dir: Path, config: dict[str, Any]) -> LoadedDataset:
    src = raw_dir / "test.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"MATH-500 raw file not found: {src}")

    rows = read_jsonl(src)
    records: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        question = first_present(row, ["problem", "question", "prompt", "text"])
        answer = first_present(row, ["answer", "solution", "target"])
        if question is None or answer is None:
            raise ValueError(f"math500 row {i}: missing question or answer; keys={sorted(row)}")
        subject = first_present(row, ["subject", "category", "type"])
        unique_id = first_present(row, ["unique_id", "id"], default=str(i))
        level = first_present(row, ["level", "difficulty"], default=None)
        try:
            difficulty = int(level) if level is not None else None
        except (TypeError, ValueError):
            difficulty = None
        records.append(
            {
                "id": f"math500_{i:04d}",
                "source_id": str(unique_id),
                "dataset": "MATH-500",
                "domain": "math",
                "question": str(question),
                "answer": str(answer),
                "difficulty": difficulty,
                "metadata": {
                    "split": "test",
                    "subject": str(subject) if subject is not None else None,
                    "level": level,
                },
            }
        )

    return LoadedDataset(
        dataset="MATH-500",
        records=records,
        row_counts={"test.jsonl": len(rows)},
        raw_columns={"test.jsonl": sorted(row.keys())},
    )
