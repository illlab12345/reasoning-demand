"""GPQA Diamond loader (dakopi/gpqa_diamond mirror, pinned revision)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from .base import LoadedDataset, make_jsonable


def _parse_query(query: str) -> tuple[str, dict[str, str]]:
    """Split 'question ... Choices:\\nA: ...\\nB: ...' into (question, options)."""
    m = re.search(r"\n?\s*Choices:\s*\n", query, flags=re.IGNORECASE)
    if not m:
        return query.strip(), {}
    question = query[: m.start()].strip()
    choices_text = query[m.end() :]
    options: dict[str, str] = {}
    for line in choices_text.splitlines():
        line = line.strip()
        mm = re.match(r"^([A-D]):\s*(.+)$", line, flags=re.IGNORECASE)
        if mm:
            options[mm.group(1).upper()] = mm.group(2).strip()
    return question, options


def load_gpqa(raw_dir: Path, config: dict[str, Any]) -> LoadedDataset:
    path = raw_dir / "data" / "train-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"GPQA raw file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    records = []
    for i, raw_row in enumerate(df.to_dict(orient="records")):
        row = make_jsonable(raw_row)
        query = str(row.get("query", ""))
        answer = str(row.get("solution", ""))
        if not query or not answer:
            raise ValueError(f"GPQA row {i}: missing query/solution; keys={sorted(row)}")
        question, options = _parse_query(query)
        records.append(
            {
                "id": f"gpqa_{i:03d}",
                "source_id": str(i),
                "dataset": "GPQA-Diamond",
                "domain": "science",
                "question": question,
                "answer": answer.upper(),
                "difficulty": None,
                "metadata": {"options": options, "raw_query": query},
            }
        )
    return LoadedDataset(
        dataset="GPQA-Diamond",
        records=records,
        row_counts={"data/train-00000-of-00001.parquet": len(df)},
        raw_columns={"data/train-00000-of-00001.parquet": sorted(df.columns)},
    )

