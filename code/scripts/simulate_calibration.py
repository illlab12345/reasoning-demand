#!/usr/bin/env python
"""Simulate the small-sample calibration protocol on the development set (0 API).

Question: if a new task family arrives with no prior demand estimates, how many
labeled items must be measured (low/high/max) before a safe allocation can be
deployed to the remaining items of the same family?

Design (exploratory, dev-set simulation; not a second prospective validation):
  * family-level variant: draw K items, estimate family-level r*_eps, apply to
    the remaining items of that family;
  * stratum-level variant: draw K items proportionally across strata, estimate
    per-stratum r*_eps (fallback: always high when a stratum is not sampled),
    apply per stratum to the remaining items.
Metrics on the holdout: mean accuracy difference vs always-high, one-sided 95%
item-level bootstrap lower bound, non-inferiority pass rate at epsilon=3pp, and
total reasoning-token saving. Averages over R=200 calibration draws.

Input : results/tables/flash_sample_level.csv (530 dev items, all settings)
Output: results/tables/calibration_simulation.json
"""

from __future__ import annotations

import csv
import json
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "tables" / "flash_sample_level.csv"
OUT = ROOT / "data" / "tables" / "calibration_simulation.json"
EPS = 0.03
SETTINGS = ["low", "high", "max"]
R_DRAWS = 200
BOOT_N = 400
SEED0 = 20260818


def _r_star(acc: dict[str, float], rt: dict[str, float], eps: float = EPS) -> str:
    a_star = max(acc.values())
    qualified = [s for s in SETTINGS if acc[s] >= a_star - eps]
    return min(qualified, key=lambda s: rt[s])


def _safe(s: str, rt: dict[str, float], threshold: float = 0.05) -> str:
    """Keep high unless the calibrated setting saves >= 5% tokens."""
    if rt.get("high") and 1.0 - rt[s] / rt["high"] >= threshold:
        return s
    return "high"


def _one_sided_lo(diffs: np.ndarray, rng: np.random.Generator, n: int = BOOT_N) -> float:
    if len(diffs) < 2:
        return float(np.mean(diffs)) if len(diffs) else 0.0
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 5.0))


def _evaluate(holdout: list[dict], r_star_by_item: dict[str, str]) -> dict:
    if not holdout:
        return None
    diffs = []
    r_tok = h_tok = 0.0
    for it in holdout:
        s = r_star_by_item[it["sample_id"]]
        diffs.append(it[f"acc_{s}"] - it["acc_high"])
        r_tok += it[f"rt_{s}"]
        h_tok += it["rt_high"]
    d = np.asarray(diffs)
    return {
        "n_holdout": len(holdout),
        "acc_diff": float(d.mean()),
        "token_saving": 1.0 - r_tok / h_tok if h_tok else 0.0,
        "diffs": d,
    }


def main() -> int:
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    families: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        fam = r["dataset"]
        families[fam].append(
            {
                "sample_id": r["sample_id"],
                "stratum": r["stratum"],
                **{f"acc_{s}": float(r[f"acc_{s}"]) for s in SETTINGS},
                **{f"rt_{s}": float(r[f"rt_{s}"]) for s in SETTINGS},
            }
        )

    strata_of = {
        "math500": [str(i) for i in range(1, 6)],
        "easy2hard_amc": [str(i) for i in range(1, 6)],
        "zebralogic_grid": [str(i) for i in range(1, 6)],
        "aime": ["1"],
        "gpqa_diamond": ["1"],
        "livecodebench": ["easy", "medium", "hard"],
    }
    k_values = [10, 20, 30]
    out: dict = {
        "epsilon": EPS,
        "design": (
            "dev-set simulation of small-sample calibration; family-level and "
            "stratum-level variants; holdout = remaining items of the same family; "
            "R=200 calibration draws; item-level one-sided bootstrap CI"
        ),
        "families": {},
    }

    for fam in families:
        items = families[fam]
        n = len(items)
        by_id = {it["sample_id"]: it for it in items}
        full_acc = {s: float(np.mean([it[f"acc_{s}"] for it in items])) for s in SETTINGS}
        full_rt = {s: float(np.mean([it[f"rt_{s}"] for it in items])) for s in SETTINGS}
        full_r = _r_star(full_acc, full_rt)
        out["families"][fam] = {"n": n, "full_r_star": full_r, "variants": {}}

        for variant in ("family", "stratum", "family_safe", "stratum_safe"):
            for k in k_values:
                if k >= n - 5:
                    continue
                safe = variant.endswith("_safe")
                draws = {"acc_diff": [], "ci_lo": [], "ni_pass": [], "token_saving": [], "r_freq": defaultdict(int)}
                for draw in range(R_DRAWS):
                    seed = SEED0 + zlib.crc32(f"{fam}|{variant}|{k}|{draw}".encode()) % (2**31)
                    rng = np.random.default_rng(seed)
                    base = variant.replace("_safe", "")
                    cal_ids: set[str] = set()
                    r_star = None
                    if base == "family":
                        cal_idx = rng.choice(n, size=k, replace=False)
                        cal_ids = {items[i]["sample_id"] for i in cal_idx}
                        cal = [items[i] for i in cal_idx]
                        cal_acc = {s: float(np.mean([it[f"acc_{s}"] for it in cal])) for s in SETTINGS}
                        cal_rt = {s: float(np.mean([it[f"rt_{s}"] for it in cal])) for s in SETTINGS}
                        r_star = _r_star(cal_acc, cal_rt)
                        if safe:
                            r_star = _safe(r_star, cal_rt)
                        r_map = {it["sample_id"]: r_star for it in items}
                    else:
                        r_map = {}
                        strata = strata_of[fam]
                        per = max(1, k // len(strata))
                        for st in strata:
                            idx = [i for i, it in enumerate(items) if it["stratum"] == st]
                            if not idx:
                                continue
                            m = min(per, len(idx))
                            chosen = list(rng.choice(idx, size=m, replace=False))
                            cal_ids.update(items[i]["sample_id"] for i in chosen)
                            if m >= 2:
                                acc = {s: float(np.mean([items[i][f"acc_{s}"] for i in chosen])) for s in SETTINGS}
                                rt = {s: float(np.mean([items[i][f"rt_{s}"] for i in chosen])) for s in SETTINGS}
                                rs = _r_star(acc, rt)
                                if safe:
                                    rs = _safe(rs, rt)
                            else:
                                rs = "high"  # safe default when a stratum is under-sampled
                            for i in idx:
                                r_map[items[i]["sample_id"]] = rs
                    if r_star is not None:
                        draws["r_freq"][r_star] += 1
                    holdout = [it for it in items if it["sample_id"] not in cal_ids]
                    res = _evaluate(holdout, r_map)
                    if res is None:
                        continue
                    draws["acc_diff"].append(res["acc_diff"])
                    draws["token_saving"].append(res["token_saving"])
                    lo = _one_sided_lo(res["diffs"], rng)
                    draws["ci_lo"].append(lo)
                    draws["ni_pass"].append(1 if lo >= -EPS else 0)

                out["families"][fam]["variants"][f"{variant}_k{k}"] = {
                    "n_draws": len(draws["acc_diff"]),
                    "acc_diff_mean": round(float(np.mean(draws["acc_diff"])), 4),
                    "acc_diff_p5": round(float(np.percentile(draws["acc_diff"], 5.0)), 4),
                    "ci_lo_median": round(float(np.median(draws["ci_lo"])), 4),
                    "ni_pass_rate": round(float(np.mean(draws["ni_pass"])), 3),
                    "token_saving_median": round(float(np.median(draws["token_saving"])), 4),
                    "r_star_freq": dict(draws["r_freq"]),
                }

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
