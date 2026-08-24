#!/usr/bin/env python
"""G2: consolidate the search-width mid experiment (0 API).

Input : results/metrics/p1_probe_20260817T150134-394d8b.jsonl and
        results/metrics/p1_probe_20260817T170415-d96244.jsonl (identical,
        cached; deduped by item/setting/rep)
Output: results/tables/searchwidth_mid.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUN_FILES = [
    ROOT / "work" / "metrics" / "p1_probe_20260817T150134-394d8b.jsonl",
    ROOT / "work" / "metrics" / "p1_probe_20260817T170415-d96244.jsonl",
]
OUT = ROOT / "data" / "tables" / "searchwidth_mid.json"
B_LEVELS = [2, 4, 8]
SETTINGS = ["low", "high", "max"]
SEED = 20260818


def _load_rows() -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for p in RUN_FILES:
        if not p.exists():
            print(f"missing run file: {p}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                by_key.setdefault((r["item_id"], r["setting"], r["rep"]), r)
    return list(by_key.values())


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
    rows = _load_rows()
    print(f"deduped rows: {len(rows)}")
    if not rows:
        return 1

    # trial-level
    trial: dict[int, dict[str, dict]] = {b: {s: {} for s in SETTINGS} for b in B_LEVELS}
    cells: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in rows:
        b = int(r["stratum"][1:])
        cells[(b, r["setting"])].append(r)

    for (b, s), rs in cells.items():
        acc = [1.0 if r["correct"] else 0.0 for r in rs]
        rt = [float(r["reasoning_tokens"]) for r in rs]
        lo, hi = _ci_boot(acc)
        trial[b][s] = {
            "n_trials": len(rs),
            "accuracy": float(np.mean(acc)),
            "ci": [round(lo, 4), round(hi, 4)],
            "mean_rt": round(float(np.mean(rt)), 1),
            "median_rt": round(float(np.median(rt)), 1),
        }

    # item-level majority
    item_cells: dict[tuple[int, int, str], list[dict]] = defaultdict(list)  # (B, base, setting)
    for r in rows:
        base = int(r["item_id"].rsplit("_", 1)[1])
        b = int(r["stratum"][1:])
        item_cells[(b, base, r["setting"])].append(r)

    item: dict[int, dict[str, object]] = {}
    for b in B_LEVELS:
        bases = sorted({k[1] for k in item_cells if k[0] == b})
        maj = {s: {} for s in SETTINGS}
        for (bb, base, s), rs in item_cells.items():
            if bb != b:
                continue
            corrects = [r["correct"] for r in rs]
            maj[s][base] = 1.0 if sum(corrects) >= 2 else 0.0

        accs = {s: float(np.mean(list(maj[s].values()))) if maj[s] else float("nan") for s in SETTINGS}
        diff_vals = [maj["high"][base] - maj["low"][base] for base in bases]
        lo2, hi2 = _ci_boot(diff_vals)
        lo1 = float(np.percentile(
            [np.mean(np.asarray(diff_vals)[np.random.default_rng(SEED + b).integers(0, len(diff_vals), size=len(diff_vals))])
             for _ in range(5000)], 5.0))
        item[b] = {
            "n_items": len(bases),
            "accuracy": {s: round(accs[s], 4) for s in SETTINGS},
            "paired_diff_high_minus_low": round(float(np.mean(diff_vals)), 4),
            "paired_ci": [round(lo2, 4), round(hi2, 4)],
            "paired_ci_lower_onesided": round(lo1, 4),
            "n_bases_needing_high": int(sum(1 for v in diff_vals if v > 0)),
        }

    out = {
        "design": {
            "task": "directed path counting (depth k=6, 20 nodes)",
            "factor": "branching factor B",
            "levels": B_LEVELS,
            "matched_bases": 30,
            "repeats": 3,
            "settings": SETTINGS,
            "max_validation": "one variant per base (30 total)",
            "source": [str(p) for p in RUN_FILES],
        },
        "trial": {str(b): trial[b] for b in B_LEVELS},
        "item": {str(b): item[b] for b in B_LEVELS},
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    for b in B_LEVELS:
        t = trial[b]
        print(
            f"  B={b}: low {t['low']['accuracy']:.2f} ({t['low']['mean_rt']:.0f} tok), "
            f"high {t['high']['accuracy']:.2f} ({t['high']['mean_rt']:.0f} tok), "
            f"max {t['max']['accuracy']:.2f} ({t['max']['mean_rt']:.0f} tok), "
            f"item diff={item[b]['paired_diff_high_minus_low']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
