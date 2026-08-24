#!/usr/bin/env python
"""Consolidate the full prospective validation (N=300, frozen Router v3) (0 API).

Input : datasets/probe/p1_full_prospective_v1.jsonl (router assignment) and the
        two full-P1 run JSONLs (deduped by item/setting/rep).
Output: results/tables/prospective_validation.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ITEM_FILE = ROOT / "datasets" / "probe" / "p1_full_prospective_v1.jsonl"
RUN_FILES = [
    ROOT / "work" / "metrics" / "p1_probe_20260816T182118-61a54b.jsonl",
    ROOT / "work" / "metrics" / "p1_probe_20260816T192221-8e34f9.jsonl",
]
OUT = ROOT / "data" / "tables" / "prospective_validation.json"
SEED = 20260818

DOMAIN_LABELS = {
    "MATH-500": "MATH",
    "Easy2Hard-Bench": "E2H",
    "ZebraLogicBench": "Zebra",
    "LiveCodeBench": "LCB",
}


def _load_rows() -> list[dict]:
    by_key: dict[tuple, dict] = {}
    for p in RUN_FILES:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                by_key.setdefault((r["item_id"], r["setting"], r["rep"]), r)
    return list(by_key.values())


def _majority(corrects: list[bool]) -> int:
    return 1 if sum(bool(c) for c in corrects) >= 2 else 0


def _boot_diff(diffs: list[float], n_boot: int = 5000) -> tuple[float, float, float]:
    arr = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(SEED)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        means[i] = arr[idx].mean()
    lo2, hi2 = np.percentile(means, [2.5, 97.5])
    lo1 = np.percentile(means, 5.0)
    return float(lo1), float(lo2), float(hi2)


def main() -> int:
    items = [json.loads(l) for l in ITEM_FILE.read_text(encoding="utf-8").splitlines() if l.strip()]
    router_setting = {it["id"]: it["_router_setting"] for it in items}
    item_domain = {it["id"]: it["dataset"] for it in items}

    rows = _load_rows()
    by_item: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["dataset"] == "MechanismProbe":
            continue
        by_item[r["item_id"]][r["setting"]].append(r)

    domains = ["MATH-500", "Easy2Hard-Bench", "ZebraLogicBench", "LiveCodeBench"]
    out: dict = {
        "n_items": len(items),
        "epsilon": 0.03,
        "rule": "Router v3 (dev-frozen; cost-aware; savings threshold 5%)",
        "reference": "Always High",
        "domains": {},
    }

    pooled_diffs: list[float] = []
    pooled_r_tok = 0.0
    pooled_h_tok = 0.0
    pooled_identical = pooled_better = pooled_worse = 0
    pooled_assigned = defaultdict(int)

    for dom in domains:
        dom_items = [it["id"] for it in items if it["dataset"] == dom]
        diffs = []
        r_tok = h_tok = 0.0
        identical = better = worse = 0
        assigned = defaultdict(int)
        for iid in dom_items:
            rs = by_item[iid]
            setting = router_setting[iid]
            assigned[setting] += 1
            h_maj = _majority([r["correct"] for r in rs.get("high", [])])
            r_maj = _majority([r["correct"] for r in rs.get(setting, [])])
            diff = r_maj - h_maj
            diffs.append(float(diff))
            r_tok += sum(float(r["reasoning_tokens"]) for r in rs.get(setting, []))
            h_tok += sum(float(r["reasoning_tokens"]) for r in rs.get("high", []))
            if diff == 0:
                identical += 1
            elif diff > 0:
                better += 1
            else:
                worse += 1
        lo1, lo2, hi2 = _boot_diff(diffs)
        out["domains"][dom] = {
            "label": DOMAIN_LABELS[dom],
            "n_items": len(diffs),
            "acc_diff": round(float(np.mean(diffs)), 4),
            "ci_lo_onesided": round(lo1, 4),
            "ci": [round(lo2, 4), round(hi2, 4)],
            "token_saving": round(1.0 - r_tok / h_tok, 4) if h_tok else 0.0,
            "identical": identical,
            "better": better,
            "worse": worse,
            "assigned": dict(assigned),
        }
        pooled_diffs += diffs
        pooled_r_tok += r_tok
        pooled_h_tok += h_tok
        pooled_identical += identical
        pooled_better += better
        pooled_worse += worse
        for s, c in assigned.items():
            pooled_assigned[s] += c

    lo1, lo2, hi2 = _boot_diff(pooled_diffs)
    out["pooled"] = {
        "label": "Pooled",
        "n_items": len(pooled_diffs),
        "acc_diff": round(float(np.mean(pooled_diffs)), 4),
        "ci_lo_onesided": round(lo1, 4),
        "ci": [round(lo2, 4), round(hi2, 4)],
        "token_saving": round(1.0 - pooled_r_tok / pooled_h_tok, 4) if pooled_h_tok else 0.0,
        "identical": pooled_identical,
        "better": pooled_better,
        "worse": pooled_worse,
        "assigned": dict(pooled_assigned),
    }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    for k in domains + ["pooled"]:
        v = out["domains"][k] if k != "pooled" else out["pooled"]
        print(
            f"  {v['label']:6s} n={v['n_items']:3d} diff={v['acc_diff']:+.3f} "
            f"ci1s={v['ci_lo_onesided']:+.3f} save={v['token_saving']*100:5.1f}% "
            f"identical={v['identical']} better={v['better']} worse={v['worse']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
