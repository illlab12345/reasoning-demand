#!/usr/bin/env python
"""Final flash-only analysis (sample-level) + figures.

Produces:
  results/tables/flash_final_analysis.json
  results/tables/flash_sample_level.csv
  results/figures/fig1_acc_vs_tokens.png
  results/figures/fig3_arr_by_domain.png
  results/figures/fig6_flip_rates.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from reasoning_efficiency.io import read_jsonl, write_json  # noqa: E402

DATASETS = ["math500", "zebralogic_grid", "easy2hard_amc"]
SETTINGS = ["low", "high", "max"]
DS_LABELS = {"math500": "MATH-500", "zebralogic_grid": "Zebra grid", "easy2hard_amc": "E2H-AMC"}


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _majority_correct(acc: float) -> bool:
    return acc >= 0.6  # 3+ correct out of 5


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Final flash analysis.")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "full_pilot_v1.jsonl")
    ap.add_argument("--output-json", type=Path, default=ROOT / "data" / "tables" / "flash_final_analysis.json")
    ap.add_argument("--output-csv", type=Path, default=ROOT / "data" / "tables" / "flash_sample_level.csv")
    ap.add_argument("--figures-dir", type=Path, default=ROOT / "data" / "figures")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = [r for r in read_jsonl(args.data) if r["model"] == "deepseek-v4-flash" and not r["error"]]

    # sample-level aggregation: per (dataset, sample, setting)
    agg: dict[tuple, list] = {}
    for r in rows:
        agg.setdefault((r["dataset"], r["sample_id"], r["reasoning_setting"]), []).append(r)
    sample = {}
    for (ds, sid, s), v in agg.items():
        sample.setdefault((ds, sid), {})[s] = {
            "acc": sum(1 for x in v if x["correct"] is True) / len(v),
            "rt": statistics.mean(x["reasoning_tokens"] or 0 for x in v),
        }
    samples_by_ds = {ds: sorted({sid for (d, sid) in sample if d == ds}) for ds in DATASETS}

    out: dict = {}
    table = {}
    print(f"{'dataset':14} {'setting':6} {'acc':>7} {'wilson_lo':>9} {'wilson_hi':>9} {'mean_rt':>9} {'med_rt':>8} {'cost':>8}")
    for ds in DATASETS:
        table[ds] = {}
        for s in SETTINGS:
            v = [r for r in rows if r["dataset"] == ds and r["reasoning_setting"] == s]
            n = len(v)
            k = sum(1 for r in v if r["correct"] is True)
            acc = k / n
            lo, hi = _wilson_ci(k, n)
            rts = [r["reasoning_tokens"] or 0 for r in v]
            table[ds][s] = {
                "n": n,
                "correct": k,
                "accuracy": round(acc, 4),
                "wilson_ci_95": [round(lo, 4), round(hi, 4)],
                "mean_reasoning_tokens": round(statistics.mean(rts), 1),
                "median_reasoning_tokens": round(statistics.median(rts), 1),
                "cost_usd": round(sum(r["cost_usd"] or 0 for r in v), 4),
            }
            print(
                f"{DS_LABELS[ds]:14} {s:6} {acc:>7.3f} {lo:>9.3f} {hi:>9.3f} "
                f"{table[ds][s]['mean_reasoning_tokens']:>9.1f} {table[ds][s]['median_reasoning_tokens']:>8.1f} "
                f"{table[ds][s]['cost_usd']:>8.4f}"
            )
    out["population"] = table

    # MRU (per 1000 tokens)
    mru = {}
    print("\nMRU (accuracy gain per 1000 reasoning tokens):")
    for ds in DATASETS:
        a = {s: table[ds][s] for s in SETTINGS}
        d_lh = a["high"]["mean_reasoning_tokens"] - a["low"]["mean_reasoning_tokens"]
        d_hm = a["max"]["mean_reasoning_tokens"] - a["high"]["mean_reasoning_tokens"]
        mru[ds] = {
            "low_to_high": round((a["high"]["accuracy"] - a["low"]["accuracy"]) / (d_lh / 1000), 4) if d_lh > 0 else None,
            "high_to_max": round((a["max"]["accuracy"] - a["high"]["accuracy"]) / (d_hm / 1000), 4) if d_hm > 0 else None,
        }
        print(
            f"  {DS_LABELS[ds]:14} low->high: {mru[ds]['low_to_high'] if mru[ds]['low_to_high'] is not None else 'n/a':>7}  "
            f"high->max: {mru[ds]['high_to_max'] if mru[ds]['high_to_max'] is not None else 'n/a':>7}"
        )
    out["mru_per_1000_tokens"] = mru

    # flip rates (sample-level majority, n=100 per dataset)
    flips = {}
    print("\nFlip rates (sample-level, majority over 5 repeats):")
    for ds in DATASETS:
        ok = {}
        for sid in samples_by_ds[ds]:
            ok[sid] = {s: _majority_correct(sample[(ds, sid)][s]["acc"]) for s in SETTINGS}
        cells = {}
        for pair_name, (a, b) in {"low_max": ("low", "max"), "high_max": ("high", "max"), "low_high": ("low", "high")}.items():
            cwr = sum(1 for sid in ok if ok[sid][a] and not ok[sid][b])
            imp = sum(1 for sid in ok if not ok[sid][a] and ok[sid][b])
            cells[pair_name] = {"correct_at_a_wrong_at_b": cwr, "wrong_at_a_correct_at_b": imp, "n": len(ok)}
        flips[ds] = cells
        print(
            f"  {DS_LABELS[ds]:14} high->max CWR={cells['high_max']['correct_at_a_wrong_at_b']} "
            f"improve={cells['high_max']['wrong_at_a_correct_at_b']} | "
            f"low->high CWR={cells['low_high']['correct_at_a_wrong_at_b']} improve={cells['low_high']['wrong_at_a_correct_at_b']}"
        )
    out["flip_rates"] = flips

    # stratum-level MSRB + ARR (paired NI per stratum, sample-level pairs n=20 per stratum per setting? use 20 samples)
    strata_out = {}
    print("\nStratum-level (flash):")
    for ds in DATASETS:
        strata = sorted({r["stratum"] for r in rows if r["dataset"] == ds})
        ds_table = {}
        for st in strata:
            sids = [sid for sid in samples_by_ds[ds] if any(r["dataset"] == ds and r["sample_id"] == sid and r["stratum"] == st for r in rows)]
            if not sids:
                continue
            acc_s, rt_s = {}, {}
            for s in SETTINGS:
                acc_s[s] = statistics.mean(sample[(ds, sid)][s]["acc"] for sid in sids)
                rt_s[s] = statistics.mean(sample[(ds, sid)][s]["rt"] for sid in sids)
            msrb = "max"
            for s in ("low", "high"):
                diffs = [sample[(ds, sid)][s]["acc"] - sample[(ds, sid)]["max"]["acc"] for sid in sids]
                m = statistics.mean(diffs)
                se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
                if m - 1.645 * se >= -0.03:
                    msrb = s
                    break
            arr = 1 - rt_s[msrb] / rt_s["max"] if rt_s["max"] else None
            ds_table[st] = {
                "n_samples": len(sids),
                "acc": {s: round(acc_s[s], 3) for s in SETTINGS},
                "mean_rt": {s: round(rt_s[s], 1) for s in SETTINGS},
                "msrb": msrb,
                "arr": round(arr, 4) if arr is not None else None,
            }
            print(
                f"  {DS_LABELS[ds]:14} stratum {st}: acc " + " ".join(f"{s}={acc_s[s]:.2f}" for s in SETTINGS)
                + f" msrb={msrb} ARR={arr:.1%}"
            )
        strata_out[ds] = ds_table
    out["stratum"] = strata_out

    write_json(args.output_json, out)

    # sample-level CSV
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "dataset", "stratum", "acc_low", "acc_high", "acc_max", "rt_low", "rt_high", "rt_max"])
        for ds in DATASETS:
            for sid in samples_by_ds[ds]:
                st = next((r["stratum"] for r in rows if r["dataset"] == ds and r["sample_id"] == sid), "")
                s = sample[(ds, sid)]
                w.writerow([sid, ds, st, *[round(s[x]["acc"], 3) for x in SETTINGS], *[round(s[x]["rt"], 1) for x in SETTINGS]])
    print(f"saved -> {args.output_csv}")

    # figures
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    colors = {"low": "#4C72B0", "high": "#DD8452", "max": "#C44E52"}
    markers = {"low": "o", "high": "s", "max": "^"}

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for ds in DATASETS:
        xs, ys, err_lo, err_hi = [], [], [], []
        for s in SETTINGS:
            t = table[ds][s]
            xs.append(t["mean_reasoning_tokens"])
            ys.append(t["accuracy"])
            lo, hi = t["wilson_ci_95"]
            err_lo.append(ys[-1] - lo)
            err_hi.append(hi - ys[-1])
        ax.errorbar(xs, ys, yerr=[err_lo, err_hi], marker="o", label=DS_LABELS[ds], capsize=4)
        for s, x, y in zip(SETTINGS, xs, ys):
            ax.annotate(s, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Mean reasoning tokens")
    ax.set_ylabel("Accuracy (Wilson 95% CI)")
    ax.set_title("Flash: Performance–Compute Frontier")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures_dir / "fig1_acc_vs_tokens.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ds_labels = [DS_LABELS[ds] for ds in DATASETS]
    arrs = []
    for ds in DATASETS:
        v = [x["arr"] for x in out["stratum"][ds].values() if x.get("arr") is not None]
        arrs.append(statistics.mean(v) if v else 0.0)
    bars = ax.bar(ds_labels, arrs, color=["#4C72B0", "#DD8452", "#C44E52"])
    for b, v in zip(bars, arrs):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.0%}", ha="center", fontsize=10)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Mean stratum-level ARR")
    ax.set_title("Flash: Avoidable Reasoning Ratio by Domain")
    fig.tight_layout()
    fig.savefig(args.figures_dir / "fig3_arr_by_domain.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = [DS_LABELS[ds] for ds in DATASETS]
    cwr = [flips[ds]["high_max"]["correct_at_a_wrong_at_b"] for ds in DATASETS]
    imp = [flips[ds]["high_max"]["wrong_at_a_correct_at_b"] for ds in DATASETS]
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], cwr, width=0.4, label="correct@high → wrong@max", color="#C44E52")
    ax.bar([i + 0.2 for i in x], imp, width=0.4, label="wrong@high → correct@max", color="#4C72B0")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Samples (of 100)")
    ax.set_title("Flash: Correct-to-Wrong Flip Rate (high vs max)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.figures_dir / "fig6_flip_rates.png", dpi=150)
    plt.close(fig)

    print(f"saved -> {args.output_json}")
    print("figures ->", args.figures_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
