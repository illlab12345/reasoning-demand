#!/usr/bin/env python
"""Development-set endpoints for the frozen Router v3 rule (0 API).

Input : results/tables/flash_sample_level.csv (item-level dev accuracy/tokens)
        results/tables/router_v3_rule.json (frozen rule)
Output: results/tables/dev_router_v3.json
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "tables" / "flash_sample_level.csv"
RULE = ROOT / "data" / "tables" / "router_v3_rule.json"
OUT = ROOT / "data" / "tables" / "dev_router_v3.json"
SEED = 42


def main() -> int:
    rule = json.loads(RULE.read_text(encoding="utf-8"))["rule"]
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    per: dict[str, list[float]] = {}
    diffs = []
    tr = th = 0.0
    assigned: dict[str, int] = {}
    for r in rows:
        key = f"{r['dataset']}|{r['stratum']}"
        s = rule[key]
        ar = float(r[f"acc_{s}"])
        ah = float(r["acc_high"])
        rr = float(r[f"rt_{s}"])
        rh = float(r["rt_high"])
        diffs.append(ar - ah)
        tr += rr
        th += rh
        per.setdefault(r["dataset"], []).append(ar - ah)
        assigned[s] = assigned.get(s, 0) + 1

    d = np.asarray(diffs)
    rng = np.random.default_rng(SEED)
    boot = np.array([d[rng.integers(0, len(d), size=len(d))].mean() for _ in range(5000)])
    out = {
        "n_items": len(d),
        "reference": "Always High",
        "pooled": {
            "acc_diff": round(float(d.mean()), 4),
            "ci_lo_onesided": round(float(np.percentile(boot, 5.0)), 4),
            "ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
            "token_reduction": round(1.0 - tr / th, 4),
            "assigned": assigned,
        },
        "per_domain": {
            k: {"n": len(v), "acc_diff": round(float(np.mean(v)), 4)} for k, v in per.items()
        },
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
