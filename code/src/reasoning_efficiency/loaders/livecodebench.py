"""LiveCodeBench loader (livecodebench/code_generation_lite test.jsonl).

Private test cases are stored as pickle(zlib(base64)) per the official LCB
code; we decode with the same chain. NOTE: pickle is loaded from the pinned
official dataset artifact only (trusted source).
"""

from __future__ import annotations

import base64
import json
import pickle
import zlib
from pathlib import Path
from typing import Any

from ..io import read_jsonl
from .base import LoadedDataset


def decode_private_tests(s: str) -> list[dict[str, Any]]:
    raw = zlib.decompress(base64.b64decode(s.encode("utf-8")))
    obj = pickle.loads(raw)
    if isinstance(obj, str):
        return json.loads(obj)
    return obj


def load_livecodebench(raw_dir: Path, config: dict[str, Any]) -> LoadedDataset:
    path = raw_dir / "test.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"LiveCodeBench raw file not found: {path}")
    rows = read_jsonl(path)
    records = []
    skipped = 0
    for i, row in enumerate(rows):
        title = row.get("question_title", "")
        content = row.get("question_content", "")
        if not content:
            skipped += 1
            continue
        starter = row.get("starter_code") or ""
        question = content if not starter else content + "\n\nStarter code:\n```python\n" + starter + "\n```"
        try:
            public = json.loads(row.get("public_test_cases", "[]"))
            private = decode_private_tests(row["private_test_cases"]) if row.get("private_test_cases") else []
        except Exception as e:  # noqa: BLE001 - malformed rows are skipped and counted
            skipped += 1
            continue
        answer = json.dumps({"public": public, "private": private}, ensure_ascii=False, sort_keys=True)
        records.append(
            {
                "id": f"lcb_{i:04d}",
                "source_id": str(row.get("question_id", i)),
                "dataset": "LiveCodeBench",
                "domain": "coding",
                "question": question,
                "answer": answer,
                "difficulty": None,
                "metadata": {
                    "question_id": row.get("question_id"),
                    "question_title": title,
                    "platform": row.get("platform"),
                    "contest_id": row.get("contest_id"),
                    "contest_date": row.get("contest_date"),
                    "difficulty": row.get("difficulty"),
                    "starter_code": starter,
                    "n_public_tests": len(public),
                    "n_private_tests": len(private),
                },
            }
        )
    return LoadedDataset(
        dataset="LiveCodeBench",
        records=records,
        row_counts={"test.jsonl": len(rows)},
        skipped_rows={"test.jsonl": skipped},
    )

