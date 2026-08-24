"""AIME 2024 loader (HuggingFaceH4/aime_2024, pinned revision)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .base import LoadedDataset, first_present, make_jsonable


def load_aime(raw_dir: Path, config: dict[str, Any]) -> LoadedDataset:
    path = raw_dir / "data" / "train-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"AIME raw file not found: {path}")
    df = pd.read_parquet(path, engine="pyarrow")
    records = []
    for i, raw_row in enumerate(df.to_dict(orient="records")):
        row = make_jsonable(raw_row)
        question = first_present(row, ["problem", "question"])
        answer = first_present(row, ["answer"])
        if question is None or answer is None:
            raise ValueError(f"AIME row {i}: missing problem/answer; keys={sorted(row)}")
        records.append(
            {
                "id": f"aime_{i:03d}",
                "source_id": str(first_present(row, ["id"], default=i)),
                "dataset": "AIME",
                "domain": "math",
                "question": str(question),
                "answer": str(answer),
                "difficulty": None,
                "metadata": {
                    "year": first_present(row, ["year"], default=None),
                    "url": first_present(row, ["url"], default=None),
                    "solution": first_present(row, ["solution"], default=None),
                },
            }
        )
    return LoadedDataset(
        dataset="AIME",
        records=records,
        row_counts={"data/train-00000-of-00001.parquet": len(df)},
        raw_columns={"data/train-00000-of-00001.parquet": sorted(df.columns)},
    )

