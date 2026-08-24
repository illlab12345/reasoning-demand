#!/usr/bin/env python
"""Router v3: token-cost-aware stratum rule with savings threshold (zero API).

Rule per (dataset, stratum):
1. ref = max setting accuracy; qualifying = {s : acc_s >= ref - epsilon}.
2. Among qualifiers, compute predicted saving vs Always High from dev mean RT.
3. Pick the qualifier with the lowest dev mean RT only if its saving >= 0.05
   (i.e., deviate from high only when we predict >=5% token savings);
   otherwise default to high.

Also simulates the rule on the 30 prospective probe items using the v2 probe's
observed tokens/accuracy, so we can decide whether a third API probe is worth it.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402

from reasoning_efficiency.io import write_json  # noqa: E402

SETTINGS = ["low", "high", "max"]
EPSILON = 0.03
SAVING_THRESHOLD = 0.05
ORDER = {"low": 0, "high": 1, "max": 2}
DATASET_MAP = {
    "MATH-500": "math500",
    "Easy2Hard-Bench": "easy2hard_amc",
    "ZebraLogicBench": "zebralogic_grid",
    "LiveCodeBench": "livecodebench",
}


def _boot_lo(values: list[float], seed: int = 7, iters: int = 2000) -> float:
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(iters))
    return round(boots[int(0.05 * iters)], 3)


def _build_rule(df: pd.DataFrame) -> dict:
    rule = {}
    for (ds, st), g in df.groupby(["dataset", "stratum"]):
        acc = {s: g[f"acc_{s}"].mean() for s in SETTINGS}
        rt = {s: g[f"rt_{s}"].mean() for s in SETTINGS}
        ref = max(acc.values())
        qual = [s for s in SETTINGS if acc[s] >= ref - EPSILON]
        if not qual:
            rule[f"{ds}|{st}"] = "high"
            continue
        best_saving = 0.0
        choice = "high"
        for s in qual:
            saving = 1 - rt[s] / rt["high"] if rt["high"] else 0.0
            if saving >= SAVING_THRESHOLD and (saving, -ORDER[s]) > (best_saving, -ORDER[choice]):
                best_saving = saving
                choice = s
        rule[f"{ds}|{st}"] = choice
    return rule


def main() -> int:
    df = pd.read_csv(ROOT / "data" / "tables" / "flash_sample_level.csv")
    rule = _build_rule(df)

    # dev evaluation
    settings_map = {}
    for _, r in df.iterrows():
        settings_map[(r["dataset"], r["sample_id"])] = rule[f"{r['dataset']}|{r['stratum']}"]

    def eval_policy(sel: dict, ref_name: str) -> dict:
        rows = []
        for _, r in df.iterrows():
            s = sel[(r["dataset"], r["sample_id"])]
            rows.append((r[f"acc_{s}"], r[f"rt_{s}"], r[f"acc_{ref_name}"]))
        diffs = [x[0] - x[2] for x in rows]
        return {
            "accuracy": round(statistics.mean(x[0] for x in rows), 3),
            "mean_rt": round(statistics.mean(x[1] for x in rows), 1),
            f"diff_vs_{ref_name}": round(statistics.mean(diffs), 3),
            f"diff_vs_{ref_name}_ci_lo": _boot_lo(diffs),
        }

    dev_eval = {"vs_high": eval_policy(settings_map, "high"), "vs_max": eval_policy(settings_map, "max")}

    # prospective simulation on the v2 probe items (30 items, observed tokens/acc)
    rows = []
    per_run = ROOT / "work" / "metrics"
    for f in sorted(per_run.glob("p1_probe_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("dataset") != "MechanismProbe" and not r.get("error"):
                rows.append(r)
    by_item = {}
    for r in rows:
        by_item.setdefault((r["dataset"], r["item_id"], r["stratum"]), {})[r["setting"]] = {
            "ok": r["correct"],
            "rt": r["reasoning_tokens"],
        }
    diffs, reds = [], []
    assigned = {}
    for (ds, item, st), v in by_item.items():
        if "high" not in v:
            continue
        key = f"{DATASET_MAP.get(ds, ds)}|{st}"
        choice = rule.get(key, "high")
        assigned[choice] = assigned.get(choice, 0) + 1
        ref = v["high"]
        a = v.get(choice, ref)
        diffs.append(a["ok"] - ref["ok"])
        reds.append((ref["rt"] - a["rt"]) / max(1, ref["rt"]))
    sim = {
        "n_items": len(diffs),
        "acc_diff": round(statistics.mean(diffs), 3),
        "acc_diff_ci_lo": _boot_lo(diffs),
        "token_reduction": round(statistics.mean(reds), 4),
        "assigned": assigned,
    }

    out = {
        "epsilon": EPSILON,
        "saving_threshold": SAVING_THRESHOLD,
        "rule": rule,
        "dev_evaluation": dev_eval,
        "prospective_simulation_on_probe_v2": sim,
    }
    write_json(ROOT / "data" / "tables" / "router_v3_rule.json", out)
    print("rule:", json.dumps(rule, ensure_ascii=False))
    print("dev:", json.dumps(dev_eval, ensure_ascii=False))
    print("prospective simulation:", json.dumps(sim, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

