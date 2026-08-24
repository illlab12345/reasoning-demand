#!/usr/bin/env python
"""Extract ZebraLogic grid complexity features and quintile strata.

Reads processed zebralogic records (grid_mode only), computes:
  - n_houses (rows), n_attributes (header columns - 1), n_cells
  - n_clues (numbered clue lines in the puzzle text)
and assigns deterministic complexity quintiles (Q1 easiest .. Q5 hardest)
based on n_cells, tie-broken by n_clues.

Outputs:
  results/tables/zebra_complexity.csv
  results/tables/zebra_complexity_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402

from reasoning_efficiency.io import read_jsonl, write_json  # noqa: E402


def _n_clues(puzzle: str) -> int:
    if not puzzle:
        return 0
    return len(re.findall(r"(?m)^\s*\d+\s*\.\s", puzzle))


def build_features(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        meta = rec.get("metadata", {})
        if meta.get("mode") != "grid":
            continue
        solution = meta.get("solution") or {}
        header = solution.get("header") or []
        grid_rows = solution.get("rows") or []
        n_attributes = max(0, len(header) - 1)
        n_houses = len(grid_rows)
        n_cells = n_houses * n_attributes
        rows.append(
            {
                "id": rec["id"],
                "source_id": rec["source_id"],
                "grid_size": meta.get("grid_size", ""),
                "n_houses": n_houses,
                "n_attributes": n_attributes,
                "n_cells": n_cells,
                "n_clues": _n_clues(rec.get("question", "")),
            }
        )
    return pd.DataFrame(rows)


def assign_quintiles(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic quintiles: rank by (n_cells, n_clues), then 5 equal-ish groups."""
    if df.empty:
        return df
    df = df.copy()
    df["complexity_rank"] = df[["n_cells", "n_clues"]].apply(
        lambda r: (int(r["n_cells"]), int(r["n_clues"])), axis=1
    ).rank(method="first", ascending=True).astype(int)
    n = len(df)
    bin_size = -(-n // 5)
    df["quintile"] = ((df["complexity_rank"] - 1) // bin_size).clip(0, 4) + 1
    return df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ZebraLogic grid complexity features.")
    ap.add_argument("--processed", type=Path, default=ROOT / "datasets" / "processed" / "zebralogic.jsonl")
    ap.add_argument("--output-csv", type=Path, default=ROOT / "data" / "tables" / "zebra_complexity.csv")
    ap.add_argument("--output-json", type=Path, default=ROOT / "data" / "tables" / "zebra_complexity_summary.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.processed.exists():
        raise SystemExit(f"processed file not found: {args.processed}")
    records = read_jsonl(args.processed)
    df = build_features(records)
    df = assign_quintiles(df)
    if df.empty:
        raise SystemExit("no grid_mode records found")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_grid_records": len(df),
        "quintile_counts": df["quintile"].value_counts().sort_index().to_dict(),
        "feature_stats": {
            col: {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": round(float(df[col].mean()), 3),
            }
            for col in ("n_houses", "n_attributes", "n_cells", "n_clues")
        },
    }
    write_json(args.output_json, summary)

    print(f"grid records: {len(df)}")
    print(df.groupby("quintile")[["n_cells", "n_clues"]].agg(["min", "max", "mean"]).round(2).to_string())
    print(f"saved -> {args.output_csv}")
    print(f"saved -> {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

