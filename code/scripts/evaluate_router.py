#!/usr/bin/env python
"""Evaluate reasoning routers on flash data (no new API calls).

Methods:
  - baselines: Always Low / High / Max
  - stratum_router: rule (in-sample stratum-level MSRB via paired NI)
  - ml_router_cv: logistic regression, 5-fold CV (within-domain)
  - ml_router_lodo: leave-one-dataset-out (cross-domain)
  - oracles: per-sample best / minimal label (upper bounds)

Metrics: accuracy (mean per-sample accuracy), mean reasoning tokens, cost,
latency, token reduction vs Always Max, accuracy loss vs Always Max.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold, StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from reasoning_efficiency.io import write_json  # noqa: E402

SETTINGS = ["low", "high", "max"]
FEATURES = ["dataset", "stratum", "question_len", "difficulty", "n_clues", "n_cells"]
DOMAIN_LABELS = {
    "math500": "MATH-500",
    "zebralogic_grid": "Zebra grid",
    "easy2hard_amc": "E2H-AMC",
    "aime": "AIME",
    "gpqa_diamond": "GPQA Diamond",
    "livecodebench": "LiveCodeBench",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate routers on flash data.")
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "processed" / "flash_router_dataset.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "tables" / "router_evaluation.json")
    ap.add_argument("--output-csv", type=Path, default=ROOT / "data" / "tables" / "router_evaluation.csv")
    return ap.parse_args()


def _stratum_msrb(df: pd.DataFrame, ds: str, stratum: str) -> str:
    sub = df[(df["dataset"] == ds) & (df["stratum"] == stratum)]
    if len(sub) < 5:
        return "max"
    for s in ("low", "high"):
        diffs = sub[f"acc_{s}"] - sub["acc_max"]
        m = diffs.mean()
        se = diffs.std(ddof=1) / math.sqrt(len(diffs))
        if m - 1.645 * se >= -0.03:
            return s
    return "max"


def _metrics(df: pd.DataFrame, settings: pd.Series) -> dict:
    acc = sum(df.loc[i, f"acc_{s}"] for i, s in settings.items()) / len(df)
    rt = sum(df.loc[i, f"rt_{s}"] for i, s in settings.items()) / len(df)
    cost = sum(df.loc[i, f"cost_{s}"] for i, s in settings.items()) / len(df)
    lat = sum(df.loc[i, f"latency_{s}"] for i, s in settings.items()) / len(df)
    rt_max = df["rt_max"].mean()
    acc_max = df["acc_max"].mean()
    return {
        "accuracy": round(acc, 4),
        "mean_reasoning_tokens": round(rt, 1),
        "mean_cost_usd": round(cost, 6),
        "mean_latency_ms": round(lat, 1),
        "token_reduction_vs_max": round(1 - rt / rt_max, 4),
        "accuracy_loss_vs_max": round(acc_max - acc, 4),
    }


def _domain_metrics(df: pd.DataFrame, settings: pd.Series, ds: str) -> dict:
    idx = df.index[df["dataset"] == ds]
    sub = df.loc[idx]
    s_sub = settings.loc[idx]
    base = _metrics(sub, s_sub)
    for ref in ("max", "high"):
        diffs = [sub.loc[i, f"acc_{s_sub.loc[i]}"] - sub.loc[i, f"acc_{ref}"] for i in idx]
        m = statistics.mean(diffs)
        se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        base[f"diff_vs_{ref}"] = round(m, 4)
        base[f"diff_vs_{ref}_ci_lo"] = round(m - 1.645 * se, 4)
    return base


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.data)
    df["dataset_id"] = df["dataset"].astype("category").cat.codes
    df["stratum_id"] = df["stratum"].astype("category").cat.codes
    X = df[["dataset_id", "stratum_id", "question_len", "difficulty", "n_clues", "n_cells"]]
    y = df["label"]

    method_settings: dict[str, pd.Series] = {}

    # fixed methods
    for s in SETTINGS:
        method_settings[f"always_{s}"] = pd.Series(s, index=df.index)

    # stratum router (in-sample rule)
    pred = {}
    for (ds, st), _ in df.groupby(["dataset", "stratum"]):
        msrb = _stratum_msrb(df, ds, st)
        idx = df[(df["dataset"] == ds) & (df["stratum"] == st)].index
        for i in idx:
            pred[i] = msrb
    method_settings["stratum_router"] = pd.Series(pred)

    # oracle upper bounds
    method_settings["oracle_best"] = df["best_setting"]
    method_settings["oracle_minimal"] = df["label"]

    # ML router: 5-fold CV within-domain
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    preds = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        clf.fit(X.iloc[tr], y.iloc[tr])
        for i, p in zip(te, clf.predict(X.iloc[te])):
            preds[i] = p
    method_settings["ml_router_cv"] = pd.Series(preds)

    # ML router: leave-one-dataset-out
    preds_lodo = {}
    gkf = GroupKFold(n_splits=len(df["dataset"].unique()))
    for tr, te in gkf.split(X, y, groups=df["dataset"]):
        clf.fit(X.iloc[tr], y.iloc[tr])
        for i, p in zip(te, clf.predict(X.iloc[te])):
            preds_lodo[i] = p
    method_settings["ml_router_lodo"] = pd.Series(preds_lodo)

    results = {name: _metrics(df, s) for name, s in method_settings.items()}
    per_domain: dict[str, dict[str, dict]] = {}
    print(f"\n{'method':18} {'domain':14} {'acc':>6} {'mean_rt':>9} {'red_vs_max':>9} {'diff_vs_max':>10} {'diff_vs_high':>11}")
    for name, s in method_settings.items():
        for ds in sorted(df["dataset"].unique()):
            m = _domain_metrics(df, s, ds)
            per_domain.setdefault(name, {})[ds] = m
            print(
                f"{name:18} {ds:14} {m['accuracy']:>6.3f} {m['mean_reasoning_tokens']:>9.1f} "
                f"{m['token_reduction_vs_max']:>9.1%} {m['diff_vs_max']:>+10.3f} {m['diff_vs_high']:>+11.3f}"
            )

    write_json(args.output, {"n_samples": len(df), "methods": results, "per_domain": per_domain})
    pd.DataFrame(results).T.to_csv(args.output_csv)
    print(f"{'method':18} {'acc':>6} {'mean_rt':>9} {'cost':>8} {'lat':>7} {'red_vs_max':>9} {'acc_loss':>8}")
    for k, v in results.items():
        print(
            f"{k:18} {v['accuracy']:>6.3f} {v['mean_reasoning_tokens']:>9.1f} {v['mean_cost_usd']:>8.5f} "
            f"{v['mean_latency_ms']:>7.0f} {v['token_reduction_vs_max']:>9.1%} {v['accuracy_loss_vs_max']:>+8.3f}"
        )

    # figure: accuracy vs tokens per domain
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    domains = sorted(df["dataset"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), sharey=True)
    axes = axes.flatten()
    marker = {"always_low": "o", "always_high": "s", "always_max": "^", "stratum_router": "D", "ml_router_lodo": "P", "oracle_minimal": "*"}
    colors = {"always_low": "#4C72B0", "always_high": "#DD8452", "always_max": "#C44E52", "stratum_router": "#55A868", "ml_router_lodo": "#8172B3", "oracle_minimal": "#CCB974"}
    order = ["always_low", "always_high", "always_max", "stratum_router", "ml_router_lodo", "oracle_minimal"]
    for ax, ds in zip(axes, domains):
        for m in order:
            if m not in per_domain:
                continue
            v = per_domain[m][ds]
            ax.scatter(v["mean_reasoning_tokens"], v["accuracy"], marker=marker[m], s=70, color=colors[m], label=m.replace("_", " "), zorder=3)
        ax.set_xlabel("Mean reasoning tokens")
        ax.set_title(DOMAIN_LABELS[ds])
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Flash: Router Trade-off (Accuracy vs Reasoning Tokens)")
    fig.tight_layout()
    fig_path = ROOT / "data" / "figures" / "fig_router_tradeoff.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {fig_path}")
    print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
