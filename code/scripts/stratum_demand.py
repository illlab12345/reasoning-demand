#!/usr/bin/env python
"""G1: stratum-level sufficient reasoning demand r*_eps (0 API).

Criterion (narrative C*_eps): for each stratum, let A* be the best
observed accuracy across {low, high, max}; a setting is qualified if
its accuracy is within epsilon of A*; r*_eps is the qualified setting
with the lowest observed mean reasoning tokens.

Input : results/tables/paper_tables.json (single source of truth)
Output: results/tables/stratum_demand.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "tables" / "paper_tables.json"
OUT = ROOT / "data" / "tables" / "stratum_demand.json"

SETTINGS = ["low", "high", "max"]
RANK = {"low": 1, "high": 2, "max": 3}


def _r_star(acc: dict, rt: dict, eps: float) -> tuple[str, list[str], float]:
    a_star = max(acc.values())
    qualified = [s for s in SETTINGS if acc[s] >= a_star - eps]
    best = min(qualified, key=lambda s: rt[s])
    return best, qualified, a_star


def main() -> int:
    d = json.loads(SRC.read_text(encoding="utf-8"))
    eps = float(d["epsilon"])
    stratum = d["stratum"]
    population = d["population"]

    out = {
        "epsilon": eps,
        "criterion": "A* = max observed acc; qualified = {s: acc_s >= A* - eps}; "
        "r*_eps = argmin mean reasoning tokens among qualified",
        "strata": {},
        "benchmark": {},
        "matrix": {},
    }

    for bench, cells in stratum.items():
        out["strata"][bench] = {}
        for s, cell in cells.items():
            acc = {k: cell["acc"][k] for k in SETTINGS}
            rt = {k: cell["rt"][k] for k in SETTINGS}
            r_star, qualified, a_star = _r_star(acc, rt, eps)
            saving = max(0.0, 1.0 - rt[r_star] / rt["max"]) if rt["max"] else 0.0
            out["strata"][bench][s] = {
                "n": cell["n"],
                "acc": {k: round(v, 4) for k, v in acc.items()},
                "mean_rt": {k: round(v, 2) for k, v in rt.items()},
                "a_star": round(a_star, 4),
                "qualified": qualified,
                "r_star": r_star,
                "r_star_rank": RANK[r_star],
                "saving_vs_max": round(saving, 4),
            }

    for bench, cell in population.items():
        acc = {k: cell[k]["accuracy"] for k in SETTINGS}
        rt = {k: cell[k]["mean_rt"] for k in SETTINGS}
        r_star, qualified, a_star = _r_star(acc, rt, eps)
        saving = max(0.0, 1.0 - rt[r_star] / rt["max"]) if rt["max"] else 0.0
        out["benchmark"][bench] = {
            "n_items": cell["low"]["n_items"],
            "a_star": round(a_star, 4),
            "qualified": qualified,
            "r_star": r_star,
            "r_star_rank": RANK[r_star],
            "saving_vs_max": round(saving, 4),
        }

    # Fig. 2 heatmap matrix (S1..S5 for MATH/E2H/Zebra, easy/med/hard for LCB)
    out["matrix"] = {
        "math500": [out["strata"]["math500"][s]["r_star"] for s in ["1", "2", "3", "4", "5"]],
        "easy2hard_amc": [out["strata"]["easy2hard_amc"][s]["r_star"] for s in ["1", "2", "3", "4", "5"]],
        "zebralogic_grid": [out["strata"]["zebralogic_grid"][s]["r_star"] for s in ["1", "2", "3", "4", "5"]],
        "livecodebench": [
            out["strata"]["livecodebench"]["easy"]["r_star"],
            out["strata"]["livecodebench"]["medium"]["r_star"],
            out["strata"]["livecodebench"]["hard"]["r_star"],
        ],
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT} ({len(out['strata'])} benchmarks, epsilon={eps})")
    for bench in ("math500", "easy2hard_amc", "zebralogic_grid", "livecodebench"):
        print(f"  {bench}: {out['matrix'][bench]}")
    print("  benchmark r*:", {k: v["r_star"] for k, v in out["benchmark"].items()})
    return 0


if __name__ == "__main__":
    sys.exit(main())
