#!/usr/bin/env python
"""P0: single source of truth for paper Tables 0-7 (frozen, flash only).

Conventions (documented once, used everywhere):
- Inferential unit = problem/item (5 repetitions are technical replicates).
- Accuracy in policy tables = fraction of items with majority-correct (>=3/5).
- Tokens/cost/latency = per-item mean over repetitions, averaged over items.
- CIs are item-clustered bootstrap percentiles (seed 42, 2000 iters).
- Non-inferiority: epsilon=0.03 vs a pre-specified reference (we report both
  vs max and vs high; MSRB rule uses max as the frozen protocol reference).

Output: results/tables/paper_tables.json
"""

from __future__ import annotations

import json
import math
import random
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

from reasoning_efficiency.io import load_yaml, read_jsonl, write_json  # noqa: E402

SETTINGS = ["low", "high", "max"]
DATASETS = ["math500", "easy2hard_amc", "zebralogic_grid", "aime", "gpqa_diamond", "livecodebench"]
EPSILON = 0.03
BOOT_ITERS = 2000


def _boot_ci(values: list[float], seed: int, alpha: float = 0.05) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(values)
    boots = []
    for _ in range(BOOT_ITERS):
        boots.append(statistics.mean(rng.choices(values, k=n)))
    boots.sort()
    lo = boots[int((alpha / 2) * BOOT_ITERS)]
    hi = boots[int((1 - alpha / 2) * BOOT_ITERS) - 1]
    return round(lo, 3), round(hi, 3)


def _load_sample_level() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "tables" / "flash_sample_level.csv")
    usage: dict[tuple, dict] = {}
    for line in open(ROOT / "data" / "processed" / "full_pilot_v1.jsonl", encoding="utf-8"):
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
    return df


def _load_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    pilot_cfg = load_yaml(ROOT / "code" / "configs" / "pilot_v1.yaml")
    comp = pd.read_csv(ROOT / "data" / "tables" / "zebra_complexity.csv")
    comp_map = {r["id"]: r for r in comp.to_dict(orient="records")}
    samples: dict[tuple, dict] = {}
    for dkey, dcfg in pilot_cfg["datasets"].items():
        p = Path(dcfg.get("pilot", ""))
        if not p.exists():
            continue
        for s in read_jsonl(p):
            samples[(dkey, s["id"])] = s
    rows = []
    for _, r in df.iterrows():
        rec = samples.get((r["dataset"], r["sample_id"]))
        if rec is None:
            continue
        ds = r["dataset"]
        if ds == "math500":
            diff = (float(rec.get("difficulty") or 3) / 5.0)
        elif ds == "easy2hard_amc":
            diff = float(rec.get("difficulty") or 0.5)
        elif ds == "zebralogic_grid":
            diff = int(comp_map.get(r["sample_id"], {}).get("n_cells", 18)) / 36.0
        elif ds == "livecodebench":
            diff = {"easy": 1 / 3, "medium": 2 / 3, "hard": 1.0}.get(str(r["stratum"]), 0.5)
        else:
            diff = 0.5
        rows.append(
            {
                "dataset": ds,
                "sample_id": r["sample_id"],
                "stratum": r["stratum"],
                "question_len": len(rec["question"]),
                "difficulty": diff,
                "n_clues": int(comp_map.get(r["sample_id"], {}).get("n_clues", 0)) if ds == "zebralogic_grid" else 0,
                "n_cells": int(comp_map.get(r["sample_id"], {}).get("n_cells", 0)) if ds == "zebralogic_grid" else 0,
            }
        )
    feats = pd.DataFrame(rows).set_index(["dataset", "sample_id"])
    labels = {}
    for (ds, sid), s in feats.iterrows():
        row = df[(df["dataset"] == ds) & (df["sample_id"] == sid)].iloc[0]
        best = max(SETTINGS, key=lambda x: (row[f"acc_{x}"], -row[f"rt_{x}"]))
        cand = [x for x in SETTINGS if row[f"acc_{x}"] >= row[f"acc_{best}"] - 0.1]
        labels[(ds, sid)] = min(cand, key=lambda x: row[f"rt_{x}"])
    return feats, pd.Series(labels)


def _stratum_msrb(sub: pd.DataFrame, stratum: str) -> str:
    g = sub[sub["stratum"] == stratum]
    if len(g) < 5:
        return "max"
    for s in ("low", "high"):
        diffs = (g[f"acc_{s}"] - g["acc_max"]).tolist()
        m = statistics.mean(diffs)
        se = statistics.stdev(diffs) / math.sqrt(len(diffs))
        if m - 1.645 * se >= -EPSILON:
            return s
    return "max"


def _policy_metrics(df: pd.DataFrame, settings: dict[tuple, str], ds: str | None = None) -> dict:
    sub = df[df["dataset"] == ds] if ds else df
    rows = []
    for _, r in sub.iterrows():
        s = settings[(r["dataset"], r["sample_id"])]
        rows.append(
            {
                "acc": r[f"acc_{s}"],
                "rt": r[f"rt_{s}"],
                "cost": r[f"cost_{s}"],
                "lat": r[f"lat_{s}"],
            }
        )
    acc = statistics.mean(x["acc"] for x in rows)
    rt = statistics.mean(x["rt"] for x in rows)
    cost = statistics.mean(x["cost"] for x in rows)
    lat = statistics.mean(x["lat"] for x in rows)
    return {
        "accuracy": round(acc, 3),
        "mean_rt": round(rt, 1),
        "mean_cost_usd": round(cost, 6),
        "mean_latency_ms": round(lat, 1),
    }


def _diff_ci(df: pd.DataFrame, settings: dict[tuple, str], ds: str, ref: str, seed: int) -> tuple[float, float]:
    sub = df[df["dataset"] == ds]
    diffs = []
    for _, r in sub.iterrows():
        s = settings[(r["dataset"], r["sample_id"])]
        diffs.append(r[f"acc_{s}"] - r[f"acc_{ref}"])
    lo, hi = _boot_ci(diffs, seed)
    return round(statistics.mean(diffs), 3), lo


def main() -> int:
    df = _load_sample_level()
    feats, labels = _load_features(df)
    out: dict = {"n_samples": len(df), "n_trials": len(df) * len(SETTINGS) * 5, "epsilon": EPSILON}

    # Table 1: population accuracy (item-clustered bootstrap CI) + tokens
    pop = {}
    med = {}
    for line in open(ROOT / "data" / "processed" / "full_pilot_v1.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r["model"] != "deepseek-v4-flash" or r["error"]:
            continue
        med.setdefault((r["dataset"], r["reasoning_setting"]), []).append(r["reasoning_tokens"] or 0)
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        pop[ds] = {}
        for s in SETTINGS:
            accs = sub[f"acc_{s}"].tolist()
            acc = statistics.mean(accs)
            ci = _boot_ci(accs, seed=1000 + len(pop) * 3 + SETTINGS.index(s))
            pop[ds][s] = {
                "n_items": len(sub),
                "accuracy": round(acc, 3),
                "ci": ci,
                "mean_rt": round(sub[f"rt_{s}"].mean(), 1),
                "median_rt": round(statistics.median(med.get((ds, s), [0])), 1),
            }
    out["population"] = pop

    # Table 2: paired NI vs max (item-level bootstrap CI) + flips
    ni = {}
    for ds in DATASETS:
        sub = df[df["dataset"] == ds].reset_index(drop=True)
        ok = {s: sub[f"acc_{s}"] >= 0.6 for s in SETTINGS}
        res = {}
        for a in ("low", "high"):
            diffs = (sub[f"acc_{a}"] - sub["acc_max"]).tolist()
            lo, _ = _boot_ci(diffs, seed=2000 + DATASETS.index(ds) * 2 + SETTINGS.index(a))
            res[a] = {
                "diff": round(statistics.mean(diffs), 3),
                "ci_lo": lo,
                "ni": lo >= -EPSILON,
                "save": round(sub["rt_max"].mean() - sub[f"rt_{a}"].mean(), 1),
                "flips": [int((ok[a] & ~ok["max"]).sum()), int((~ok[a] & ok["max"]).sum())],
            }
        ni[ds] = res
    out["ni"] = ni

    # Table 3: MRU
    out["mru"] = {}
    for ds in DATASETS:
        p = pop[ds]
        d1 = p["high"]["mean_rt"] - p["low"]["mean_rt"]
        d2 = p["max"]["mean_rt"] - p["high"]["mean_rt"]
        out["mru"][ds] = {
            "low_high": round((p["high"]["accuracy"] - p["low"]["accuracy"]) / (d1 / 1000), 4) if d1 > 0 else None,
            "high_max": round((p["max"]["accuracy"] - p["high"]["accuracy"]) / (d2 / 1000), 4) if d2 > 0 else None,
        }

    # Table 4: stratum MSRB / ARR
    out["stratum"] = {}
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        st = {}
        for stt in sorted(sub["stratum"].unique()):
            g = sub[sub["stratum"] == stt]
            rt = {s: round(g[f"rt_{s}"].mean(), 1) for s in SETTINGS}
            acc = {s: round(g[f"acc_{s}"].mean(), 3) for s in SETTINGS}
            msrb = _stratum_msrb(sub, stt)
            st[stt] = {"n": len(g), "acc": acc, "rt": rt, "msrb": msrb, "arr": round(1 - rt[msrb] / rt["max"], 4)}
        out["stratum"][ds] = st

    # Table 6: instance-level waste
    out["waste"] = {}
    pooled = {"low->high": [0, 0], "low->max": [0, 0], "high->max": [0, 0]}  # [n, total]
    for ds in DATASETS:
        sub = df[df["dataset"] == ds].reset_index(drop=True)
        ok = {s: sub[f"acc_{s}"] >= 0.6 for s in SETTINGS}
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
            pooled[f"{a}->{b}"][0] += int(m.sum())
            pooled[f"{a}->{b}"][1] += diff.sum()
    out["waste_pooled"] = {
        k: {"n": v[0], "total": round(v[1], 1), "mean": round(v[1] / v[0], 1) if v[0] else None}
        for k, v in pooled.items()
    }

    # Table 5: router policies (majority-based accuracy; per-item mean tokens)
    settings: dict[tuple, str] = {}
    for s in SETTINGS:
        for (ds, sid) in labels.index:
            settings[(ds, sid)] = s
    strat_settings = {}
    for ds in DATASETS:
        sub = df[df["dataset"] == ds]
        for stt in sorted(sub["stratum"].unique()):
            msrb = _stratum_msrb(sub, stt)
            for (d2, sid) in labels.index:
                if d2 == ds and df[(df["dataset"] == ds) & (df["sample_id"] == sid)]["stratum"].iloc[0] == stt:
                    strat_settings[(ds, sid)] = msrb
    oracle = {}
    for (ds, sid) in labels.index:
        row = df[(df["dataset"] == ds) & (df["sample_id"] == sid)].iloc[0]
        oracle[(ds, sid)] = next((s for s in SETTINGS if row[f"acc_{s}"] >= 0.6), "low")

    feats_flat = feats.reset_index()
    X = feats_flat[["dataset", "stratum", "question_len", "difficulty", "n_clues", "n_cells"]].copy()
    X["dataset_id"] = X["dataset"].astype("category").cat.codes
    X["stratum_id"] = X["stratum"].astype("category").cat.codes
    X = X[["dataset_id", "stratum_id", "question_len", "difficulty", "n_clues", "n_cells"]]
    y = labels
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    pred_cv, pred_lodo = {}, {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        clf.fit(X.iloc[tr], y.iloc[tr])
        for i, p in zip(te, clf.predict(X.iloc[te])):
            pred_cv[labels.index[i]] = p
    gkf = GroupKFold(n_splits=len(X["dataset_id"].unique()))
    for tr, te in gkf.split(X, y, groups=X["dataset_id"]):
        clf.fit(X.iloc[tr], y.iloc[tr])
        for i, p in zip(te, clf.predict(X.iloc[te])):
            pred_lodo[labels.index[i]] = p

    methods = {
        "always_low": {k: "low" for k in labels.index},
        "always_high": {k: "high" for k in labels.index},
        "always_max": {k: "max" for k in labels.index},
        "stratum_router": strat_settings,
        "ml_router_cv": pred_cv,
        "ml_router_lodo": pred_lodo,
        "oracle": oracle,
    }
    out["router_global"] = {}
    for name, mset in methods.items():
        out["router_global"][name] = _policy_metrics(df, mset)
    maxg = out["router_global"]["always_max"]
    for name, m in out["router_global"].items():
        m["token_reduction_vs_max"] = round(1 - m["mean_rt"] / maxg["mean_rt"], 4)
        m["acc_diff_vs_max"] = round(m["accuracy"] - maxg["accuracy"], 4)

    out["router_per_domain"] = {}
    for name, mset in methods.items():
        out["router_per_domain"][name] = {ds: _policy_metrics(df, mset, ds) for ds in DATASETS}
    for name in methods:
        for ds in DATASETS:
            m = out["router_per_domain"][name][ds]
            base = out["router_per_domain"]["always_max"][ds]
            m["token_reduction_vs_max"] = round(1 - m["mean_rt"] / base["mean_rt"], 4)
            m["acc_diff_vs_max"] = round(m["accuracy"] - base["accuracy"], 4)
    for name in ("stratum_router", "ml_router_lodo"):
        for ds in DATASETS:
            diff, lo = _diff_ci(df, methods[name], ds, "max", seed=3000 + DATASETS.index(ds) + (0 if name.startswith("stratum") else 100))
            out["router_per_domain"][name][ds]["diff_vs_max"] = diff
            out["router_per_domain"][name][ds]["diff_vs_max_ci_lo"] = lo
            diff_h, lo_h = _diff_ci(df, methods[name], ds, "high", seed=4000 + DATASETS.index(ds) + (0 if name.startswith("stratum") else 100))
            out["router_per_domain"][name][ds]["diff_vs_high"] = diff_h
            out["router_per_domain"][name][ds]["diff_vs_high_ci_lo"] = lo_h

    # Table 7: oracle end-to-end (same basis as Table 5)
    orc = out["router_global"]["oracle"]
    totals = {
        "mean_rt": round(orc["mean_rt"] * len(df), 1),
        "mean_cost_usd": round(orc["mean_cost_usd"] * len(df), 6),
    }
    out["oracle"] = {
        "accuracy": orc["accuracy"],
        "total_reasoning_tokens": totals["mean_rt"],
        "total_cost_usd": totals["mean_cost_usd"],
        "savings": {},
    }
    for ref in ("high", "max"):
        base = out["router_global"][f"always_{ref}"]
        out["oracle"]["savings"][f"vs_always_{ref}"] = {
            "token_reduction": round(1 - orc["mean_rt"] / base["mean_rt"], 4),
            "cost_reduction": round(1 - orc["mean_cost_usd"] / base["mean_cost_usd"], 4),
            "accuracy_delta": round(orc["accuracy"] - base["accuracy"], 4),
        }

    write_json(ROOT / "data" / "tables" / "paper_tables.json", out)
    print(json.dumps({k: v for k, v in out.items() if k in ("n_samples", "n_trials")}, ensure_ascii=False))
    print("router_global:", json.dumps(out["router_global"], ensure_ascii=False, indent=1))
    print("oracle:", json.dumps(out["oracle"], ensure_ascii=False, indent=1))
    print("saved -> results/tables/paper_tables.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
