#!/usr/bin/env python
"""Build flash router dataset: per-sample features + empirical outcomes + labels.

Label (instance-level minimal setting): the lowest reasoning setting whose
sample accuracy (mean over 5 repeats) is within 0.1 of the best setting,
tie-broken by lower mean reasoning tokens.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402

from reasoning_efficiency.io import load_yaml, read_jsonl, write_json  # noqa: E402

SETTINGS = ["low", "high", "max"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build flash router dataset.")
    ap.add_argument("--sample-csv", type=Path, default=ROOT / "data" / "tables" / "flash_sample_level.csv")
    ap.add_argument("--full-pilot", type=Path, default=ROOT / "data" / "processed" / "full_pilot_v1.jsonl")
    ap.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "pilot_v1.yaml")
    ap.add_argument("--complexity", type=Path, default=ROOT / "data" / "tables" / "zebra_complexity.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "flash_router_dataset.csv")
    ap.add_argument("--output-json", type=Path, default=ROOT / "data" / "tables" / "flash_router_dataset_summary.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    pilot_cfg = load_yaml(args.config)
    samples: dict[str, dict[str, dict]] = {}
    for dkey, dcfg in pilot_cfg["datasets"].items():
        for s in read_jsonl(Path(dcfg["pilot"])):
            samples.setdefault(dkey, {})[s["id"]] = s

    complexity = pd.read_csv(args.complexity)
    comp_map = {r["id"]: r for r in complexity.to_dict(orient="records")} if len(complexity) else {}

    # per-sample per-setting cost/latency
    usage: dict[tuple, dict] = {}
    for r in read_jsonl(args.full_pilot):
        if r["model"] != "deepseek-v4-flash" or r["error"]:
            continue
        key = (r["dataset"], r["sample_id"], r["reasoning_setting"])
        u = usage.setdefault(key, {"cost": [], "latency": []})
        if r["cost_usd"] is not None:
            u["cost"].append(r["cost_usd"])
        if r["latency_ms"] is not None:
            u["latency"].append(r["latency_ms"])

    df = pd.read_csv(args.sample_csv)
    rows = []
    for _, row in df.iterrows():
        ds = row["dataset"]
        sid = row["sample_id"]
        rec = samples[ds][sid]
        if ds == "math500":
            difficulty_norm = (float(rec.get("difficulty") or 3) / 5.0)
        elif ds == "easy2hard_amc":
            difficulty_norm = float(rec.get("difficulty") or 0.5)
        elif ds == "zebralogic_grid":
            difficulty_norm = int(comp_map.get(sid, {}).get("n_cells", 18)) / 36.0
        elif ds == "livecodebench":
            difficulty_norm = {"easy": 1 / 3, "medium": 2 / 3, "hard": 1.0}.get(str(row["stratum"]), 0.5)
        else:  # aime / gpqa_diamond (no official difficulty labels)
            difficulty_norm = 0.5
        features = {
            "dataset": ds,
            "stratum": str(row["stratum"]),
            "question_len": len(rec["question"]),
            "difficulty": difficulty_norm,
            "n_clues": int(comp_map.get(sid, {}).get("n_clues", 0)) if ds == "zebralogic_grid" else 0,
            "n_cells": int(comp_map.get(sid, {}).get("n_cells", 0)) if ds == "zebralogic_grid" else 0,
        }
        accs = {s: float(row[f"acc_{s}"]) for s in SETTINGS}
        rts = {s: float(row[f"rt_{s}"]) for s in SETTINGS}
        costs = {s: statistics.mean(usage.get((ds, sid, s), {}).get("cost") or [0.0]) for s in SETTINGS}
        lats = {s: statistics.mean(usage.get((ds, sid, s), {}).get("latency") or [0.0]) for s in SETTINGS}
        best = max(SETTINGS, key=lambda s: (accs[s], -rts[s]))
        candidates = [s for s in SETTINGS if accs[s] >= accs[best] - 0.1]
        label = min(candidates, key=lambda s: rts[s])
        rows.append(
            {
                "sample_id": sid,
                **features,
                **{f"acc_{s}": accs[s] for s in SETTINGS},
                **{f"rt_{s}": rts[s] for s in SETTINGS},
                **{f"cost_{s}": round(costs[s], 6) for s in SETTINGS},
                **{f"latency_{s}": round(lats[s], 1) for s in SETTINGS},
                "best_setting": best,
                "label": label,
            }
        )

    out = pd.DataFrame(rows)
    out.to_csv(args.output, index=False)
    summary = {
        "n_samples": len(out),
        "by_dataset": out["dataset"].value_counts().to_dict(),
        "label_distribution": out["label"].value_counts().to_dict(),
        "best_setting_distribution": out["best_setting"].value_counts().to_dict(),
    }
    write_json(args.output_json, summary)
    print(f"wrote {len(out)} rows -> {args.output}")
    print("labels:", summary["label_distribution"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
