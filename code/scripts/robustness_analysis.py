#!/usr/bin/env python
"""Robustness analyses for the paper (zero API calls).

1. epsilon sensitivity (1/3/5pp) of best-attainable-reference MSRB/ARR.
2. Heavy-tail token statistics: P50/P90/P95/P99, top-10% share, Gini.
3. Micro vs macro accuracy (pooled vs equal-weight across benchmarks).
4. Competence-matched analysis: Spearman correlation between model
   competence proxies (acc_low / acc_high / solvable fraction) and ARR,
   at benchmark level and at stratum level.

Output: results/tables/robustness_analysis.json
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from reasoning_efficiency.io import read_jsonl, write_json  # noqa: E402

SETTINGS = ["low", "high", "max"]
DATASETS = ["math500", "easy2hard_amc", "zebralogic_grid", "aime", "gpqa_diamond", "livecodebench"]


def _gini(values: list[float]) -> float:
    vals = sorted(values)
    n = len(vals)
    if n == 0 or sum(vals) == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(vals))
    return round((2 * cum) / (n * sum(vals)) - (n + 1) / n, 4)


def _top10_share(values: list[float]) -> float:
    vals = sorted(values, reverse=True)
    total = sum(vals)
    if total == 0:
        return 0.0
    k = max(1, len(vals) // 10)
    return round(sum(vals[:k]) / total, 4)


def main() -> int:
    df = pd.read_csv(ROOT / "data" / "tables" / "flash_sample_level.csv")
    out: dict = {}

    # ---- 1. epsilon sensitivity (dataset-level, best-attainable reference) ----
    eps_sens = {}
    for eps in (0.01, 0.03, 0.05):
        row = {}
        for ds in DATASETS:
            sub = df[df["dataset"] == ds]
            acc = {s: sub[f"acc_{s}"].mean() for s in SETTINGS}
            rt = {s: sub[f"rt_{s}"].mean() for s in SETTINGS}
            ref = max(acc.values())
            qual = [s for s in SETTINGS if acc[s] >= ref - eps]
            msrb = min(qual, key=lambda s: rt[s]) if qual else "max"
            row[ds] = {"msrb": msrb, "arr": round(1 - rt[msrb] / rt["max"], 4) if rt["max"] else None}
        eps_sens[str(eps)] = row
    out["epsilon_sensitivity"] = eps_sens

    # ---- 2. heavy-tail token statistics (per benchmark, pooled over settings) ----
    rows = [r for r in read_jsonl(ROOT / "data" / "processed" / "full_pilot_v1.jsonl") if r["model"] == "deepseek-v4-flash" and not r["error"]]
    heavy = {}
    for ds in DATASETS:
        vals = [r["reasoning_tokens"] or 0 for r in rows if r["dataset"] == ds]
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        q = lambda p: vals_sorted[min(n - 1, int(p * n))]
        heavy[ds] = {
            "n": n,
            "P50": round(q(0.50), 1),
            "P90": round(q(0.90), 1),
            "P95": round(q(0.95), 1),
            "P99": round(q(0.99), 1),
            "top10_share": _top10_share(vals),
            "gini": _gini(vals),
        }
    out["heavy_tail_tokens"] = heavy

    # ---- 3. micro vs macro accuracy per setting ----
    micro_macro = {}
    for s in SETTINGS:
        micro = sum(1 for r in rows if r["reasoning_setting"] == s and r["correct"] is True) / sum(
            1 for r in rows if r["reasoning_setting"] == s
        )
        per_ds = []
        for ds in DATASETS:
            v = [r for r in rows if r["dataset"] == ds and r["reasoning_setting"] == s]
            per_ds.append(sum(1 for r in v if r["correct"] is True) / len(v))
        micro_macro[s] = {"micro": round(micro, 4), "macro": round(statistics.mean(per_ds), 4)}
    out["micro_macro"] = micro_macro

    # ---- 4. competence-matched analysis ----
    arr_bench = {
        "math500": 0.506,
        "easy2hard_amc": 0.360,
        "zebralogic_grid": 0.0,
        "aime": 0.530,
        "gpqa_diamond": 0.0,
        "livecodebench": 0.685,
    }
    bench = []
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        bench.append(
            {
                "dataset": ds,
                "acc_low": round(sub["acc_low"].mean(), 3),
                "acc_high": round(sub["acc_high"].mean(), 3),
                "solvable_frac": round((sub[["acc_low", "acc_high", "acc_max"]].max(axis=1) >= 0.6).mean(), 3),
                "arr": arr_bench[ds],
            }
        )
    bench_df = pd.DataFrame(bench)
    bench_corr = {}
    for proxy in ("acc_low", "acc_high", "solvable_frac"):
        rho, p = spearmanr(bench_df[proxy], bench_df["arr"])
        bench_corr[proxy] = {"spearman": round(rho, 3), "p": round(p, 4)}
    out["competence_matched_benchmark"] = {"table": bench, "spearman": bench_corr}

    # stratum level
    strata = []
    paper = json.loads((ROOT / "data" / "tables" / "paper_tables.json").read_text(encoding="utf-8"))["stratum"]
    for ds in DATASETS:
        for st, v in paper[ds].items():
            acc_high = v["acc"]["high"]
            arr = v["arr"]
            strata.append({"ds": ds, "stratum": st, "acc_high": acc_high, "arr": arr})
    strata_df = pd.DataFrame(strata)
    rho_s, p_s = spearmanr(strata_df["acc_high"], strata_df["arr"])
    out["competence_matched_stratum"] = {
        "n": len(strata_df),
        "spearman_acc_high_vs_arr": {"spearman": round(rho_s, 3), "p": round(p_s, 4)},
    }

    write_json(ROOT / "data" / "tables" / "robustness_analysis.json", out)
    print("=== epsilon sensitivity (MSRB / ARR) ===")
    for eps in ("0.01", "0.03", "0.05"):
        print(eps, json.dumps(eps_sens[eps], ensure_ascii=False))
    print("=== heavy tail (pooled per benchmark) ===")
    for ds, v in heavy.items():
        print(ds, json.dumps(v))
    print("=== micro/macro ===")
    print(json.dumps(micro_macro, ensure_ascii=False))
    print("=== competence-matched ===")
    print("benchmark:", json.dumps(bench_corr, ensure_ascii=False))
    print("stratum:", json.dumps(out["competence_matched_stratum"], ensure_ascii=False))
    print("saved -> results/tables/robustness_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

