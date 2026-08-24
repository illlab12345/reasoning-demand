#!/usr/bin/env python
"""Router v2: best-attainable-reference rule (zero API calls).

For each (dataset, stratum): ref = max setting accuracy; a setting qualifies if
its accuracy >= ref - epsilon (3pp); among qualifiers pick the one with the
lowest observed mean reasoning tokens (tie: low < high < max).

Also evaluates the rule on development data (item-level bootstrap CI vs Always
High / Always Max) and compares with an ML CV router. Saves the frozen rule.
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
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from reasoning_efficiency.io import load_yaml, read_jsonl, write_json  # noqa: E402

SETTINGS = ["low", "high", "max"]
EPSILON = 0.03
ORDER = {"low": 0, "high": 1, "max": 2}


def _boot_ci(values: list[float], seed: int, iters: int = 2000) -> float:
    rng = random.Random(seed)
    n = len(values)
    boots = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(iters))
    return round(boots[int(0.05 * iters)], 3)


def main() -> int:
    df = pd.read_csv(ROOT / "data" / "tables" / "flash_sample_level.csv")
    old_cfg = load_yaml(ROOT / "code" / "configs" / "p1_probe.yaml")
    old_rule = old_cfg.get("frozen_router_rule", {})

    rule = {}
    for (ds, st), g in df.groupby(["dataset", "stratum"]):
        acc = {s: g[f"acc_{s}"].mean() for s in SETTINGS}
        rt = {s: g[f"rt_{s}"].mean() for s in SETTINGS}
        ref = max(acc.values())
        qual = [s for s in SETTINGS if acc[s] >= ref - EPSILON]
        choice = min(qual, key=lambda s: (rt[s], ORDER[s])) if qual else "high"
        rule[f"{ds}|{st}"] = choice

    changes = {k: (old_rule.get(k), rule.get(k)) for k in sorted(set(old_rule) | set(rule)) if old_rule.get(k) != rule.get(k)}
    print("rule changes vs v1:", json.dumps(changes, ensure_ascii=False, indent=1))

    # evaluate rule on dev
    settings_map = {}
    for _, r in df.iterrows():
        settings_map[(r["dataset"], r["sample_id"])] = rule[f"{r['dataset']}|{r['stratum']}"]

    def eval_policy(sel: dict, ref_name: str) -> dict:
        rows = []
        for _, r in df.iterrows():
            s = sel[(r["dataset"], r["sample_id"])]
            rows.append((r[f"acc_{s}"], r[f"rt_{s}"], r[f"acc_{ref_name}"]))
        acc = statistics.mean(x[0] for x in rows)
        rt = statistics.mean(x[1] for x in rows)
        diffs = [x[0] - x[2] for x in rows]
        return {
            "accuracy": round(acc, 3),
            "mean_rt": round(rt, 1),
            f"diff_vs_{ref_name}": round(statistics.mean(diffs), 3),
            f"diff_vs_{ref_name}_ci_lo": _boot_ci(diffs, seed=11),
        }

    dev_eval = {
        "vs_high": eval_policy(settings_map, "high"),
        "vs_max": eval_policy(settings_map, "max"),
    }

    # ML CV comparison on dev
    feats = pd.DataFrame(
        {
            "dataset_id": df["dataset"].astype("category").cat.codes,
            "stratum_id": df["stratum"].astype("category").cat.codes,
            "question_len": 0.0,
            "difficulty": 0.5,
            "n_clues": 0.0,
            "n_cells": 0.0,
        }
    )
    labels = []
    for _, r in df.iterrows():
        best = max(SETTINGS, key=lambda s: (r[f"acc_{s}"], -r[f"rt_{s}"]))
        cand = [s for s in SETTINGS if r[f"acc_{s}"] >= r[f"acc_{best}"] - 0.1]
        labels.append(min(cand, key=lambda s: r[f"rt_{s}"]))
    labels = pd.Series(labels)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    preds = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(feats, labels):
        clf.fit(feats.iloc[tr], labels.iloc[tr])
        for i, p in zip(te, clf.predict(feats.iloc[te])):
            preds[i] = p
    ml_settings = {}
    for i, (_, r) in enumerate(df.iterrows()):
        ml_settings[(r["dataset"], r["sample_id"])] = preds[i]
    ml_eval = {"vs_high": eval_policy(ml_settings, "high"), "vs_max": eval_policy(ml_settings, "max")}

    out = {
        "epsilon": EPSILON,
        "rule": rule,
        "dev_evaluation_rule_v2": dev_eval,
        "dev_evaluation_ml_cv": ml_eval,
        "rule_changes_vs_v1": changes,
    }
    write_json(ROOT / "data" / "tables" / "router_v2_rule.json", out)
    print("dev rule v2:", json.dumps(dev_eval, ensure_ascii=False))
    print("dev ML CV:", json.dumps(ml_eval, ensure_ascii=False))
    print("saved -> results/tables/router_v2_rule.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

