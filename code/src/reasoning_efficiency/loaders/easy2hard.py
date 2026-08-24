"""Easy2Hard-Bench loader (furonghuang-lab/Easy2Hard-Bench, pinned revision).

Per-subset field mappings are explicit because each subset has its own schema:
- E2H-AMC: numeric answer (free response), difficulty = rating.
- E2H-GSM8K: numeric answer, difficulty = rating.
- E2H-Winogrande: sentence + option1/option2, answer = option index ("1"/"2").
- E2H-ARC: question + choices dict {label, text}, answer = answerKey letter.
- E2H-Codeforces: answer = JSON of expected outputs (needs execution evaluator).
- E2H-Lichess: answer = first move in SAN (needs chess evaluator).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .base import LoadedDataset, first_present, make_jsonable

DIFFICULTY_KEYS = ["rating", "difficulty", "difficulty_score", "irt_difficulty"]


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _difficulty(row: dict[str, Any]) -> float | None:
    for k in DIFFICULTY_KEYS:
        v = _to_float(first_present(row, [k]))
        if v is not None:
            return v
    return None


def _norm_arc_choices(choices: Any) -> dict[str, str] | None:
    """Convert ARC choices dict {label: [...], text: [...]} to {label: text}."""
    if not isinstance(choices, dict):
        return choices
    labels, texts = choices.get("label"), choices.get("text")
    if isinstance(labels, list) and isinstance(texts, list) and len(labels) == len(texts):
        return {str(l): str(t) for l, t in zip(labels, texts)}
    return choices


def _base_record(
    subset: str,
    domain: str,
    row: dict[str, Any],
    index: int,
    question: str,
    answer: str,
    difficulty: float | None,
    source_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"easy2hard_{subset}_{index:05d}",
        "source_id": source_id,
        "dataset": "Easy2Hard-Bench",
        "domain": domain,
        "question": question,
        "answer": answer,
        "difficulty": difficulty,
        "metadata": metadata,
    }


def _amc_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = str(first_present(row, ["problem", "question"], default=""))
    answer = str(first_present(row, ["answer"], default=""))
    if not question or not answer:
        return None
    source_id = f"{row.get('contest', '')}-{row.get('year', '')}-{row.get('month', '')}-{row.get('index', index)}"
    return _base_record(
        "E2H-AMC", "math", row, index, question, answer, _difficulty(row), source_id,
        {
            "subset": "E2H-AMC",
            "split": _split(row),
            "contest": row.get("contest"),
            "year": row.get("year"),
            "month": row.get("month"),
            "subtest": row.get("subtest"),
            "tag": row.get("tag"),
            "rating_std": row.get("rating_std"),
            "item_difficulty": row.get("item_difficulty"),
            "options": None,
        },
    )


def _gsm8k_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = str(first_present(row, ["question", "problem"], default=""))
    answer = str(first_present(row, ["answer"], default=""))
    if not question or not answer:
        return None
    return _base_record(
        "E2H-GSM8K", "math", row, index, question, answer, _difficulty(row), str(index),
        {
            "subset": "E2H-GSM8K",
            "split": _split(row),
            "rating_std": row.get("rating_std"),
            "model_avg_acc": row.get("model_avg_acc"),
            "options": None,
        },
    )


def _winogrande_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    sentence = str(first_present(row, ["sentence", "question"], default=""))
    opt1, opt2 = row.get("option1"), row.get("option2")
    answer = str(first_present(row, ["answer"], default=""))
    if not sentence or opt1 is None or opt2 is None or not answer:
        return None
    return _base_record(
        "E2H-Winogrande", "commonsense", row, index, sentence, answer, _difficulty(row), str(index),
        {
            "subset": "E2H-Winogrande",
            "split": _split(row),
            "options": [str(opt1), str(opt2)],
            "rating_std": row.get("rating_std"),
            "model_avg_acc": row.get("model_avg_acc"),
        },
    )


def _arc_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    question = str(first_present(row, ["question", "problem"], default=""))
    answer = str(first_present(row, ["answerKey", "answer"], default=""))
    choices = _norm_arc_choices(row.get("choices"))
    if not question or not answer:
        return None
    return _base_record(
        "E2H-ARC", "reasoning", row, index, question, answer, _difficulty(row), str(row.get("id", index)),
        {
            "subset": "E2H-ARC",
            "split": _split(row),
            "options": choices,
            "rating_std": row.get("rating_std"),
            "model_avg_acc": row.get("model_avg_acc"),
        },
    )


def _codeforces_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    main = str(first_present(row, ["problem_main", "problem"], default=""))
    if not main.strip():
        return None  # missing problem statement; input/output specs alone are not a valid question
    parts = [main]
    for label, key in (("Input", "input_spec"), ("Output", "output_spec")):
        v = row.get(key)
        if v:
            parts.append(f"### {label}\n{v}")
    question = "\n\n".join(parts)
    answers = row.get("answers")
    if answers is None:
        return None
    answer = json.dumps({"answers": make_jsonable(list(answers))}, ensure_ascii=False, sort_keys=True)
    source_id = f"{row.get('contest_id', '')}-{row.get('problem_index', '')}"
    return _base_record(
        "E2H-Codeforces", "coding", row, index, question, answer, _difficulty(row), source_id,
        {
            "subset": "E2H-Codeforces",
            "split": _split(row),
            "contest_id": row.get("contest_id"),
            "problem_index": row.get("problem_index"),
            "tag": row.get("tag"),
            "detailed_tag": row.get("detailed_tag"),
            "original_tags": row.get("original_tags"),
            "rating_std": row.get("rating_std"),
            "rating_volatility": row.get("rating_volatility"),
            "sample_inputs": row.get("sample_inputs"),
            "sample_outputs": row.get("sample_outputs"),
            "input_output": row.get("input_output"),
            "options": None,
        },
    )


def _lichess_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    fen = str(row.get("fen", ""))
    pgn = str(row.get("pgn", ""))
    answer = str(first_present(row, ["answer_san", "answer_uci"], default=""))
    if not fen or not answer:
        return None
    question = (
        f"Chess puzzle (puzzle_id={row.get('puzzle_id', '')}).\n"
        f"FEN: {fen}\nPGN: {pgn}\n"
        "What is the best move? Answer with the SAN notation of the first move."
    )
    return _base_record(
        "E2H-Lichess", "chess", row, index, question, answer, _difficulty(row), str(row.get("puzzle_id", index)),
        {
            "subset": "E2H-Lichess",
            "split": _split(row),
            "fen": fen,
            "pgn": pgn,
            "answer_san": row.get("answer_san"),
            "answer_uci": row.get("answer_uci"),
            "init_num_moves": row.get("init_num_moves"),
            "tag": row.get("tag"),
            "rating_std": row.get("rating_std"),
            "options": None,
        },
    )


def _split(row: dict[str, Any]) -> str:
    # split is derived from the file name by the caller; stored into metadata later
    return row.get("_split", "unknown")


def _load_parquet(
    path: Path,
    subset: str,
    domain: str,
    index_start: int,
    handler,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = pd.read_parquet(path, engine="pyarrow")
    split = "eval" if "eval" in path.name else ("train" if "train" in path.name else "unknown")
    records: list[dict[str, Any]] = []
    skipped = 0
    for i, raw_row in enumerate(df.to_dict(orient="records")):
        row = make_jsonable(raw_row)
        row["_split"] = split
        rec = handler(row, index_start + i)
        if rec is None:
            skipped += 1
            continue
        rec["metadata"]["split"] = split
        records.append(rec)
    info = {"row_count": len(df), "columns": sorted(df.columns)}
    info["skipped"] = skipped
    return records, info


def load_easy2hard(raw_dir: Path, config: dict[str, Any]) -> LoadedDataset:
    handlers = {
        "E2H-AMC": _amc_record,
        "E2H-Codeforces": _codeforces_record,
        "E2H-ARC": _arc_record,
        "E2H-GSM8K": _gsm8k_record,
        "E2H-Lichess": _lichess_record,
        "E2H-Winogrande": _winogrande_record,
    }
    subsets: dict[str, dict[str, Any]] = config.get("subsets", {})
    records: list[dict[str, Any]] = []
    row_counts: dict[str, int] = {}
    raw_columns: dict[str, list[str]] = {}
    skipped_rows: dict[str, int] = {}

    for subset_name in sorted(subsets):
        subset_cfg = subsets[subset_name]
        domain = subset_cfg["domain"]
        handler = handlers[subset_name]
        idx = 0
        for rel in subset_cfg["files"]:
            path = raw_dir / rel
            if not path.exists():
                raise FileNotFoundError(f"Easy2Hard raw file not found: {path}")
            sub_records, info = _load_parquet(path, subset_name, domain, idx, handler)
            records.extend(sub_records)
            idx += info["row_count"]
            row_counts[rel] = info["row_count"]
            raw_columns[rel] = info["columns"]
            skipped_rows[rel] = info["skipped"]

    return LoadedDataset(
        dataset="Easy2Hard-Bench",
        records=records,
        row_counts=row_counts,
        raw_columns=raw_columns,
        skipped_rows=skipped_rows,
    )
