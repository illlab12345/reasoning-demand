#!/usr/bin/env python
"""Summarize the state-tracking mechanism experiment (0 API).

Input : results/metrics/p1_probe_20260818T102103-279214.jsonl
Output: results/tables/statetrack_mid.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_FILE = ROOT / "work" / "metrics" / "p1_probe_20260818T102103-279214.jsonl"
OUT = ROOT / "data" / "tables" / "statetrack_mid.json"
K_LEVELS = [2, 4, 8]
SETTINGS = ["low", "high", "max"]
SEED = 20260818


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
    rows = [json.loads(l) for l in RUN_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"rows: {len(rows)}")
    if not rows:
        return 1

    trial: dict[int, dict[str, dict]] = {k: {s: {} for s in SETTINGS} for k in K_LEVELS}
    cells: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        k = int(r["stratum"][1:])
        cells[(k, r["setting"])].append(r)

    for (k, s), rs in cells.items():
        acc = [1.0 if r["correct"] else 0.0 for r in rs]
        rt = [float(r["reasoning_tokens"]) for r in rs]
        lo, hi = _ci_boot(acc)
        trial[k][s] = {
            "n_trials": len(rs),
            "accuracy": round(float(np.mean(acc)), 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "mean_rt": round(float(np.mean(rt)), 1),
            "median_rt": round(float(np.median(rt)), 1),
        }

    item_cells: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for r in rows:
        base = int(r["item_id"].rsplit("_", 1)[1])
        k = int(r["stratum"][1:])
        item_cells[(k, base, r["setting"])].append(r)

    item: dict[int, dict[str, object]] = {}
    for k in K_LEVELS:
        bases = sorted({b for (kk, b, _) in item_cells if kk == k})
        maj = {s: {} for s in SETTINGS}
        for (kk, b, s), rs in item_cells.items():
            if kk != k:
                continue
            maj[s][b] = 1.0 if sum(r["correct"] for r in rs) >= 2 else 0.0
        accs = {s: (round(float(np.mean(list(maj[s].values()))), 4) if maj[s] else None) for s in SETTINGS}
        diffs = [maj["high"][b] - maj["low"][b] for b in bases]
        lo2, hi2 = _ci_boot(diffs)
        rng = np.random.default_rng(SEED + k)
        lo1 = float(np.percentile(
            [np.mean(np.asarray(diffs)[rng.integers(0, len(diffs), size=len(diffs))]) for _ in range(5000)], 5.0
        ))
        item[k] = {
            "n_items": len(bases),
            "accuracy": accs,
            "paired_diff_high_minus_low": round(float(np.mean(diffs)), 4),
            "paired_ci": [round(lo2, 4), round(hi2, 4)],
            "paired_ci_lower_onesided": round(lo1, 4),
            "n_items_needing_high": int(sum(1 for v in diffs if v > 0)),
        }

    out = {
        "design": {
            "task": "parallel multi-variable tracking (6 steps, final sum)",
            "factor": "state-tracking load k (number of simultaneously tracked variables)",
            "levels": K_LEVELS,
            "matched_bases": 20,
            "repeats": 3,
            "settings": SETTINGS,
            "max_validation": "k=8 variant of every base (20 items)",
            "source": str(RUN_FILE),
        },
        "trial": {str(k): trial[k] for k in K_LEVELS},
        "item": {str(k): item[k] for k in K_LEVELS},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    for k in K_LEVELS:
        t = trial[k]
        max_str = (
            f"max {t['max']['accuracy']:.2f} ({t['max']['mean_rt']:.0f} tok)"
            if t.get("max")
            else "max n/a"
        )
        print(
            f"  k={k}: low {t['low']['accuracy']:.2f} ({t['low']['mean_rt']:.0f} tok), "
            f"high {t['high']['accuracy']:.2f} ({t['high']['mean_rt']:.0f} tok), {max_str}, "
            f"item diff={item[k]['paired_diff_high_minus_low']:+.3f} "
            f"(1s lo {item[k]['paired_ci_lower_onesided']:+.3f}, need-high {item[k]['n_items_needing_high']}/20)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
