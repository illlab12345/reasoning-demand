"""ZebraLogicBench loader.

Source: WildEval/ZebraLogic mirror (pinned revision), which contains the same
puzzles as allenai/ZebraLogicBench plus ground-truth answers:
- mc_mode: `answer` is the correct option *text* (e.g. "Bob").
- grid_mode: `solution` is a dict {header: [...], rows: [[...]]}.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .base import LoadedDataset, first_present, make_jsonable

FILES = {
    "mc": "mc_mode/test-00000-of-00001.parquet",
    "grid": "grid_mode/test-00000-of-00001.parquet",
}


def _read_parquet(path: Path) -> tuple[pd.DataFrame, int]:
    df = pd.read_parquet(path, engine="pyarrow")
    return df, len(df)


def _question_text(row: dict[str, Any], mode: str) -> str:
    q = first_present(row, ["question", "puzzle", "prompt", "text", "problem"])
    if q is None:
        raise ValueError(f"zebra {mode} row has no question; keys={sorted(row)}")
    return str(q)


def _mc_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    qid = first_present(row, ["id", "qid"], default=f"mc_{index}")
    question = _question_text(row, "mc")
    answer = first_present(row, ["answer", "correct_answer", "label"])
    choices = first_present(row, ["choices", "options"], default=[])
    if answer is None:
        raise ValueError(f"zebra mc row {qid}: missing answer; keys={sorted(row)}")
    return {
        "id": f"zebralogic_{index:05d}",
        "source_id": str(qid),
        "dataset": "ZebraLogicBench",
        "domain": "logic",
        "question": question,
        "answer": str(answer),
        "difficulty": None,
        "metadata": {
            "mode": "mc",
            "options": list(choices),
            "puzzle": str(first_present(row, ["puzzle"], default="")),
            "created_at": str(first_present(row, ["created_at"], default="")),
        },
    }


def _grid_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    qid = first_present(row, ["id", "qid"], default=f"grid_{index}")
    question = _question_text(row, "grid")
    solution = row.get("solution")
    if not solution:
        raise ValueError(f"zebra grid row {qid}: missing solution; keys={sorted(row)}")
    solution = make_jsonable(solution)
    return {
        "id": f"zebralogic_{index:05d}",
        "source_id": str(qid),
        "dataset": "ZebraLogicBench",
        "domain": "logic",
        "question": question,
        "answer": json.dumps(solution, ensure_ascii=False, sort_keys=True),
        "difficulty": None,
        "metadata": {
            "mode": "grid",
            "grid_size": str(first_present(row, ["size", "grid_size"], default="")),
            "solution": solution,
            "created_at": str(first_present(row, ["created_at"], default="")),
        },
    }


def load_zebralogic(raw_dir: Path, config: dict[str, Any]) -> LoadedDataset:
    records: list[dict[str, Any]] = []
    row_counts: dict[str, int] = {}
    raw_columns: dict[str, list[str]] = {}

    for mode, rel in FILES.items():
        path = raw_dir / rel
        if not path.exists():
            raise FileNotFoundError(f"ZebraLogic raw file not found: {path}")
        df, n = _read_parquet(path)
        row_counts[rel] = n
        raw_columns[rel] = sorted(df.columns)
        for raw_row in df.to_dict(orient="records"):
            row = make_jsonable(raw_row)
            rec = _mc_record(row, len(records)) if mode == "mc" else _grid_record(row, len(records))
            records.append(rec)

    return LoadedDataset(
        dataset="ZebraLogicBench",
        records=records,
        row_counts=row_counts,
        raw_columns=raw_columns,
    )
