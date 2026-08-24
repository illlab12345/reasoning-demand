#!/usr/bin/env python
"""Compute all numbers needed for the merged paper tables (Tables 0-7).

Outputs results/tables/merged_tables_data.json (population + Wilson CI, paired
NI + flips, MRU, stratum MSRB/ARR, instance-level waste, pooled waste).
"""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402

from reasoning_efficiency.io import write_json  # noqa: E402

DATASETS = ["math500", "easy2hard_amc", "zebralogic_grid", "aime", "gpqa_diamond", "livecodebench"]
SETTINGS = ["low", "high", "max"]


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)


def main() -> int:
    df = pd.read_csv(ROOT / "data" / "tables" / "flash_sample_level.csv")
    out: dict = {"population": {}, "ni": {}, "mru": {}, "stratum": {}, "waste": {}, "waste_pooled_6domain": {}}
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        pop = {}
        for s in SETTINGS:
            acc = sub[f"acc_{s}"].mean()
            n = len(sub) * 5
            k = round(acc * n)
            pop[s] = {"acc": round(acc, 3), "ci": _wilson(k, n), "rt": round(sub[f"rt_{s}"].mean(), 1)}
        out["population"][ds] = pop
        d1 = pop["high"]["rt"] - pop["low"]["rt"]
        d2 = pop["max"]["rt"] - pop["high"]["rt"]
        out["mru"][ds] = {
            "low_high": round((pop["high"]["acc"] - pop["low"]["acc"]) / (d1 / 1000), 4) if d1 > 0 else None,
            "high_max": round((pop["max"]["acc"] - pop["high"]["acc"]) / (d2 / 1000), 4) if d2 > 0 else None,
        }

        sub = sub.reset_index(drop=True)
        ok = {s: sub[f"acc_{s}"] >= 0.6 for s in SETTINGS}
        res = {}
        for a in ("low", "high"):
            diffs = (sub[f"acc_{a}"] - sub["acc_max"]).tolist()
            m = statistics.mean(diffs)
            se = statistics.stdev(diffs) / math.sqrt(len(diffs))
            ci_lo = m - 1.645 * se
            res[a] = {
                "diff": round(m, 3),
                "ci_lo": round(ci_lo, 3),
                "ni": ci_lo >= -0.03,
                "save": round(sub["rt_max"].mean() - sub[f"rt_{a}"].mean(), 1),
                "flips": [int((ok[a] & ~ok["max"]).sum()), int((~ok[a] & ok["max"]).sum())],
            }
        out["ni"][ds] = res

        st = {}
        for stt in sorted(sub["stratum"].unique()):
            g = sub[sub["stratum"] == stt]
            acc = {s: round(g[f"acc_{s}"].mean(), 3) for s in SETTINGS}
            rt = {s: round(g[f"rt_{s}"].mean(), 1) for s in SETTINGS}
            msrb = "max"
            for s in ("low", "high"):
                diffs = (g[f"acc_{s}"] - g["acc_max"]).tolist()
                m = statistics.mean(diffs)
                se = statistics.stdev(diffs) / math.sqrt(len(diffs))
                if m - 1.645 * se >= -0.03:
                    msrb = s
                    break
            st[stt] = {
                "n": len(g),
                "acc": acc,
                "rt": rt,
                "msrb": msrb,
                "arr": round(1 - rt[msrb] / rt["max"], 4) if rt["max"] else None,
            }
        out["stratum"][ds] = st

        tot = {}
        for a, b in (("low", "high"), ("low", "max"), ("high", "max")):
            m = ok[a] & ok[b]
            if m.sum() == 0:
                continue
            diff = sub.loc[m, f"rt_{b}"] - sub.loc[m, f"rt_{a}"]
            out["waste"].setdefault(ds, {})[f"{a}->{b}"] = {
                "n": int(m.sum()),
                "mean": round(diff.mean(), 1),
                "median": round(diff.median(), 1),
                "total": round(diff.sum(), 1),
                "overhead": round((diff / sub.loc[m, f"rt_{a}"]).mean(), 4),
            }
            tot[f"{a}->{b}"] = tot.get(f"{a}->{b}", 0) + diff.sum()
        for k, v in tot.items():
            out["waste_pooled_6domain"][k] = round(v, 1)

    write_json(ROOT / "data" / "tables" / "merged_tables_data.json", out)
    print("saved merged_tables_data.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

