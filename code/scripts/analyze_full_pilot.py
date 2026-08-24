#!/usr/bin/env python
"""Full-pilot analysis: coverage, accuracy, tokens, non-inferiority, ART/ARR,
overthinking flips, stratum-level MSRB.

Uses the consolidated dataset built by build_full_pilot_dataset.py.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

from reasoning_efficiency.io import load_yaml, read_jsonl, write_json  # noqa: E402

DATASETS = ["math500", "zebralogic_grid", "easy2hard_amc", "aime", "gpqa_diamond", "livecodebench"]
SETTINGS = ["low", "high", "max"]
EXPECTED = {
    "math500": 1500,
    "zebralogic_grid": 1500,
    "easy2hard_amc": 1500,
    "aime": 450,
    "gpqa_diamond": 1500,
    "livecodebench": 1500,
}


def _mean(xs: list) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(statistics.mean(vals), 2) if vals else None


def _mcnemar(ok_s: set, ok_max: set, n_pairs: int) -> dict:
    b = len(ok_s - ok_max)
    c = len(ok_max - ok_s)
    diff = (b - c) / n_pairs if n_pairs else 0.0
    se = math.sqrt(max(0.0, (b + c - ((b - c) ** 2) / n_pairs)) / (n_pairs ** 2)) if n_pairs else 0.0
    ci_lo = diff - 1.645 * se
    p = None
    if b + c > 0:
        from math import comb

        n_disc = b + c
        k = min(b, c)
        p = min(1.0, 2 * sum(comb(n_disc, i) * (0.5 ** n_disc) for i in range(k + 1)))
    return {
        "n_pairs": n_pairs,
        "s_only_correct": b,
        "max_only_correct": c,
        "difference": round(diff, 4),
        "ci_lower": round(ci_lo, 4),
        "mcnemar_p": round(p, 4) if p is not None else None,
        "non_inferior_at_epsilon_0_03": ci_lo >= -0.03,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Full-pilot analysis.")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "full_pilot_v1.jsonl")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "tables" / "full_pilot_analysis.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.data)
    exp = load_yaml(ROOT / "code" / "configs" / "experiment.yaml")
    epsilon = exp["epsilon"]
    out: dict = {"n_rows": len(rows), "epsilon": epsilon}

    # coverage
    coverage = {}
    for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
        for ds in DATASETS:
            n = sum(1 for r in rows if r["model"] == model and r["dataset"] == ds)
            expected = EXPECTED.get(ds, n)
            coverage[f"{model}|{ds}"] = {"n": n, "expected": expected, "coverage": round(n / expected, 4) if expected else None}
    out["coverage"] = coverage
    print("== coverage ==")
    for k, v in coverage.items():
        print(f"  {k:38} {v['n']:>4}/{v['expected']} {v['coverage']:.1%}")

    # population level
    print("\n== population (model x dataset x setting) ==")
    pop = {}
    for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
        for ds in DATASETS:
            for s in SETTINGS:
                v = [r for r in rows if r["model"] == model and r["dataset"] == ds and r["reasoning_setting"] == s and not r["error"]]
                if not v:
                    continue
                acc = sum(1 for r in v if r["correct"] is True) / len(v)
                key = f"{model}|{ds}|{s}"
                pop[key] = {
                    "n": len(v),
                    "accuracy": round(acc, 4),
                    "mean_reasoning_tokens": _mean([r["reasoning_tokens"] for r in v]),
                    "median_reasoning_tokens": round(statistics.median([r["reasoning_tokens"] or 0 for r in v]), 1),
                    "cost_usd": round(sum(r["cost_usd"] or 0 for r in v), 4),
                }
                print(f"  {key:42} n={len(v):>4} acc={acc:.3f} mean_rt={pop[key]['mean_reasoning_tokens']} cost={pop[key]['cost_usd']}")
    out["population"] = pop

    # paired non-inferiority + flips (per model x dataset, using pairs present in both settings)
    print("\n== paired analysis vs max ==")
    paired = {}
    for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
        for ds in DATASETS:
            base = {}
            for r in rows:
                if r["model"] != model or r["dataset"] != ds or r["error"]:
                    continue
                pid = (r["sample_id"], r["repetition_id"])
                base.setdefault(r["reasoning_setting"], {})[pid] = r["correct"] is True
            max_ok = base.get("max", {})
            ds_out = {}
            for s in ("low", "high"):
                ok_s = base.get(s, {})
                common = set(ok_s) & set(max_ok)
                comp = _mcnemar({k for k in common if ok_s[k]}, {k for k in common if max_ok[k]}, len(common))
                mean_rt_max = _mean([r["reasoning_tokens"] for r in rows if r["model"] == model and r["dataset"] == ds and r["reasoning_setting"] == "max"])
                mean_rt_s = _mean([r["reasoning_tokens"] for r in rows if r["model"] == model and r["dataset"] == ds and r["reasoning_setting"] == s])
                comp["token_saving_vs_max"] = round((mean_rt_max or 0) - (mean_rt_s or 0), 1)
                flips = {
                    "correct_at_s_wrong_at_max": sum(1 for k in common if ok_s[k] and not max_ok[k]),
                    "wrong_at_s_correct_at_max": sum(1 for k in common if not ok_s[k] and max_ok[k]),
                }
                comp["flips"] = flips
                ds_out[s] = comp
                print(
                    f"  {model}|{ds} {s:4} vs max: n={comp['n_pairs']:>4} diff={comp['difference']:+.3f} "
                    f"ci_lo={comp['ci_lower']:+.3f} NI={comp['non_inferior_at_epsilon_0_03']} "
                    f"save={comp['token_saving_vs_max']:>7.0f} flips={flips}"
                )
            paired[f"{model}|{ds}"] = ds_out
    out["paired_vs_max"] = paired

    # stratum-level: accuracy + tokens + MSRB (lowest setting NI vs max within stratum)
    print("\n== stratum level (n per cell) ==")
    strata_out = {}
    for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
        for ds in DATASETS:
            cells = {}
            for r in rows:
                if r["model"] != model or r["dataset"] != ds or r["error"]:
                    continue
                cells.setdefault((r["stratum"], r["reasoning_setting"]), []).append(r)
            strata = sorted({r["stratum"] for r in rows if r["model"] == model and r["dataset"] == ds})
            ds_table = {}
            for st in strata:
                row_cfg = {}
                for s in SETTINGS:
                    v = cells.get((st, s), [])
                    if not v:
                        continue
                    acc = sum(1 for r in v if r["correct"] is True) / len(v)
                    row_cfg[s] = {
                        "n": len(v),
                        "accuracy": round(acc, 3),
                        "mean_reasoning_tokens": _mean([r["reasoning_tokens"] for r in v]),
                    }
                # stratum MSRB: lowest setting with enough pairs and NI vs max
                max_ok = {r["sample_id"] + f"#{r['repetition_id']}" for r in cells.get((st, "max"), []) if r["correct"] is True}
                msrb = "max"
                for s in ("low", "high"):
                    ok_s = {r["sample_id"] + f"#{r['repetition_id']}" for r in cells.get((st, s), []) if r["correct"] is True}
                    common = set(ok_s) & set(max_ok)
                    if len(common) >= 10:
                        comp = _mcnemar(ok_s, max_ok, len(common))
                        if comp["non_inferior_at_epsilon_0_03"]:
                            msrb = s
                            break
                row_cfg["msrb"] = msrb
                ds_table[st] = row_cfg
                acc_str = " ".join(f"{s}={row_cfg[s]['accuracy']:.2f}({row_cfg[s]['n']})" for s in SETTINGS if s in row_cfg)
                print(f"  {model}|{ds} stratum {st}: {acc_str} msrb={msrb}")
            strata_out[f"{model}|{ds}"] = ds_table
    out["stratum"] = strata_out

    # dataset-level ART/ARR from stratum MSRB
    print("\n== dataset-level MSRB / ART / ARR (flash complete; pro partial) ==")
    art = {}
    for model in ("deepseek-v4-flash", "deepseek-v4-pro"):
        for ds in DATASETS:
            full = [r for r in rows if r["model"] == model and r["dataset"] == ds and not r["error"]]
            if not full:
                continue
            mean_max = statistics.mean(r["reasoning_tokens"] or 0 for r in full if r["reasoning_setting"] == "max")
            # dataset MSRB = lowest setting NI vs max over all pairs
            base = {}
            for r in full:
                base.setdefault(r["reasoning_setting"], {})[(r["sample_id"], r["repetition_id"])] = r["correct"] is True
            max_ok = base.get("max", {})
            msrb = "max"
            for s in ("low", "high"):
                ok_s = base.get(s, {})
                common = set(ok_s) & set(max_ok)
                if common and _mcnemar({k for k in common if ok_s[k]}, {k for k in common if max_ok[k]}, len(common))["non_inferior_at_epsilon_0_03"]:
                    msrb = s
                    break
            mean_msrb = statistics.mean(r["reasoning_tokens"] or 0 for r in full if r["reasoning_setting"] == msrb)
            arr = 1 - mean_msrb / mean_max if mean_max else None
            art[model + "|" + ds] = {"msrb": msrb, "mean_rt_max": round(mean_max, 1), "mean_rt_msrb": round(mean_msrb, 1), "arr": round(arr, 4) if arr is not None else None}
            print(f"  {model}|{ds}: msrb={msrb} mean_rt_max={mean_max:.0f} mean_rt_msrb={mean_msrb:.0f} ARR={arr:.1%}")
    out["art_arr"] = art

    write_json(args.output, out)
    print(f"\nsaved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
