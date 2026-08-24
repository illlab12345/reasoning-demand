#!/usr/bin/env python
"""Oracle end-to-end upper bound for reasoning routing (flash).

Oracle definition: for each sample, pick the LOWEST reasoning setting that is
majority-correct (>=3/5 repetitions); if no setting is correct, use `low`.
This is a theoretical upper bound (requires ground truth / perfect predictor).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402

from reasoning_efficiency.io import write_json  # noqa: E402

SETTINGS = ["low", "high", "max"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Oracle router upper bound (flash).")
    ap.add_argument("--sample-csv", type=Path, default=ROOT / "data" / "tables" / "flash_sample_level.csv")
    ap.add_argument("--full-pilot", type=Path, default=ROOT / "data" / "processed" / "full_pilot_v1.jsonl")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "tables" / "oracle_router.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.sample_csv)

    usage: dict[tuple, dict] = {}
    for line in open(args.full_pilot, encoding="utf-8"):
        r = json.loads(line)
        if r["model"] != "deepseek-v4-flash" or r["error"]:
            continue
        key = (r["dataset"], r["sample_id"], r["reasoning_setting"])
        u = usage.setdefault(key, {"cost": [], "lat": []})
        if r["cost_usd"] is not None:
            u["cost"].append(r["cost_usd"])
        if r["latency_ms"] is not None:
            u["lat"].append(r["latency_ms"])

    for s in SETTINGS:
        df[f"cost_{s}"] = df.apply(
            lambda r: statistics.mean(usage[(r["dataset"], r["sample_id"], s)]["cost"] or [0.0]), axis=1
        )
        df[f"lat_{s}"] = df.apply(
            lambda r: statistics.mean(usage[(r["dataset"], r["sample_id"], s)]["lat"] or [0.0]), axis=1
        )
        df[f"ok_{s}"] = df[f"acc_{s}"] >= 0.6

    def pick(r: pd.Series) -> str:
        for s in SETTINGS:
            if r[f"ok_{s}"]:
                return s
        return "low"

    df["setting"] = df.apply(pick, axis=1)

    def totals(policy: str | pd.Series) -> dict:
        if isinstance(policy, str):
            mask = pd.Series(True, index=df.index)
            acc = df[f"ok_{policy}"].mean()
            rt = df[f"rt_{policy}"].sum()
            cost = df[f"cost_{policy}"].sum()
            lat = (df[f"lat_{policy}"] * 5).sum()
        else:
            mask = pd.Series(True, index=df.index)
            acc = df[["ok_low", "ok_high", "ok_max"]].any(axis=1).mean()
            rt = sum(df.loc[df["setting"] == s, f"rt_{s}"].sum() for s in SETTINGS)
            cost = sum(df.loc[df["setting"] == s, f"cost_{s}"].sum() for s in SETTINGS)
            lat = sum((df.loc[df["setting"] == s, f"lat_{s}"] * 5).sum() for s in SETTINGS)
        return {
            "accuracy": round(float(acc), 4),
            "total_reasoning_tokens": round(float(rt), 1),
            "total_cost_usd": round(float(cost), 6),
            "total_latency_ms": round(float(lat), 1),
        }

    policies = {f"always_{s}": totals(s) for s in SETTINGS}
    policies["oracle_minimal"] = totals(df["setting"])

    savings = {}
    for ref in ("high", "max"):
        base = policies[f"always_{ref}"]
        orc = policies["oracle_minimal"]
        savings[f"vs_always_{ref}"] = {
            "token_reduction": round(1 - orc["total_reasoning_tokens"] / base["total_reasoning_tokens"], 4),
            "cost_reduction": round(1 - orc["total_cost_usd"] / base["total_cost_usd"], 4),
            "latency_reduction": round(1 - orc["total_latency_ms"] / base["total_latency_ms"], 4),
            "accuracy_delta": round(orc["accuracy"] - base["accuracy"], 4),
        }

    out = {
        "n_samples": len(df),
        "policies": policies,
        "savings": savings,
        "oracle_setting_distribution": df["setting"].value_counts().to_dict(),
        "unsolvable_at_all_settings": int((~df[["ok_low", "ok_high", "ok_max"]].any(axis=1)).sum()),
    }
    write_json(args.output, out)
    for name, v in policies.items():
        print(f"{name:16} acc={v['accuracy']:.3f} rt={v['total_reasoning_tokens']:.0f} cost=${v['total_cost_usd']:.4f} lat={v['total_latency_ms']:.0f}ms")
    print("savings:", json.dumps(savings, indent=2))
    print("distribution:", out["oracle_setting_distribution"], "unsolvable:", out["unsolvable_at_all_settings"])
    print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

