#!/usr/bin/env python
"""Consolidate depth / distractor / constraint mechanism results (0 API).

Input : datasets/probe/p1_full_mechanism_v1.jsonl (item metadata) and the two
        full-P1 run JSONLs (deduped by item/setting/rep).
Output: results/tables/mechanism_summary.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ITEM_FILE = ROOT / "datasets" / "probe" / "p1_full_mechanism_v1.jsonl"
RUN_FILES = [
    ROOT / "work" / "metrics" / "p1_probe_20260816T182118-61a54b.jsonl",
    ROOT / "work" / "metrics" / "p1_probe_20260816T192221-8e34f9.jsonl",
]
OUT = ROOT / "data" / "tables" / "mechanism_summary.json"
SETTINGS = ["low", "high", "max"]
SEED = 20260818


def _level(item: dict) -> int:
    m = item["metadata"]
    if m["factor"] == "depth":
        return int(m["depth"])
    if m["factor"] == "distractor":
        return len(m.get("distractors") or [])
    if m["factor"] == "constraints":
        return int(m["n_clues"])
    raise KeyError(m["factor"])


def _ci_boot(values: list[float], alpha: float = 0.05, n_boot: int = 5000) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        means[i] = arr[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def main() -> int:
    items = [json.loads(l) for l in ITEM_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    meta = {it["id"]: (it["metadata"]["factor"], _level(it)) for it in items}

    by_key: dict[tuple, dict] = {}
    for p in RUN_FILES:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                by_key.setdefault((r["item_id"], r["setting"], r["rep"]), r)
    rows = [r for r in by_key.values() if r["dataset"] == "MechanismProbe"]
    print(f"mechanism rows: {len(rows)} / {len(by_key)}")

    cells: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    item_cells: dict[tuple[str, int, str, str], list[dict]] = defaultdict(list)
    for r in rows:
        factor, level = meta.get(r["item_id"], (None, None))
        if factor is None:
            continue
        cells[(factor, level, r["setting"])].append(r)
        item_cells[(factor, level, r["setting"], r["item_id"])].append(r)

    factors = {"depth": [], "distractor": [], "constraints": []}
    for (factor, level, setting), rs in sorted(cells.items()):
        acc = [1.0 if r["correct"] else 0.0 for r in rs]
        rt = [float(r["reasoning_tokens"]) for r in rs]
        lo, hi = _ci_boot(acc)
        factors[factor].append(
            {
                "level": level,
                "setting": setting,
                "n_trials": len(rs),
                "accuracy": round(float(np.mean(acc)), 4),
                "ci": [round(lo, 4), round(hi, 4)],
                "mean_rt": round(float(np.mean(rt)), 1),
                "median_rt": round(float(np.median(rt)), 1),
            }
        )

    # item-level paired (high - low) per factor/level
    paired: dict[tuple[str, int], dict] = {}
    for (factor, level, setting, item_id), rs in item_cells.items():
        key = (factor, level)
        d = paired.setdefault(key, {"items": set(), "low": {}, "high": {}})
        maj = 1.0 if sum(r["correct"] for r in rs) >= 2 else 0.0
        if setting == "low":
            d["low"][item_id] = maj
        elif setting == "high":
            d["high"][item_id] = maj

    item_out: dict[str, list[dict]] = {"depth": [], "distractor": [], "constraints": []}
    for (factor, level), d in sorted(paired.items()):
        common = sorted(d["low"].keys() & d["high"].keys())
        diffs = [d["high"][i] - d["low"][i] for i in common]
        if not diffs:
            continue
        lo1 = float(np.percentile(
            [np.mean(np.asarray(diffs)[np.random.default_rng(SEED).integers(0, len(diffs), size=len(diffs))])
             for _ in range(5000)], 5.0))
        item_out[factor].append(
            {
                "level": level,
                "n_items": len(common),
                "paired_diff_high_minus_low": round(float(np.mean(diffs)), 4),
                "paired_ci_lower_onesided": round(lo1, 4),
                "n_items_needing_high": int(sum(1 for v in diffs if v > 0)),
            }
        )

    out = {
        "design": {
            "items": 180,
            "levels": {"depth": [8, 16, 24], "distractor": [0, 2, 4], "constraints": [4, 8, 12]},
            "repeats": 3,
            "settings": SETTINGS,
        },
        "cells": factors,
        "item_level": item_out,
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    for factor in factors:
        for c in factors[factor]:
            print(
                f"  {factor} level={c['level']} {c['setting']}: "
                f"acc {c['accuracy']:.2f} ci[{c['ci'][0]:.2f},{c['ci'][1]:.2f}] rt {c['mean_rt']:.0f}"
            )
    for factor in item_out:
        for c in item_out[factor]:
            print(f"  paired {factor} level={c['level']}: diff {c['paired_diff_high_minus_low']:+.3f} (1s lo {c['paired_ci_lower_onesided']:+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
