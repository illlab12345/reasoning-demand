#!/usr/bin/env python
"""Instance-level wasted-token analysis (flash).

Definition: on samples where BOTH settings complete the task correctly
(majority vote over 5 repetitions, accuracy >= 0.6), the wasted tokens of the
higher setting are T_higher - T_lower. Reports mean/median/total waste and
percentage overhead per dataset and pooled.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402

from reasoning_efficiency.io import write_json  # noqa: E402

PAIRS = [("low", "high"), ("low", "max"), ("high", "max")]
DATASETS = ["math500", "zebralogic_grid", "easy2hard_amc", "aime", "gpqa_diamond", "livecodebench"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Instance-level wasted tokens (flash).")
    ap.add_argument("--sample-csv", type=Path, default=ROOT / "data" / "tables" / "flash_sample_level.csv")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "tables" / "wasted_tokens.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.sample_csv)
    out: dict = {}
    print(f"{'dataset':16} {'pair':10} {'n_shared':>8} {'mean_waste':>10} {'median_waste':>12} {'total_waste':>11} {'pct_overhead':>12}")
    for ds in DATASETS:
        sub = df[df["dataset"] == ds].copy()
        ok = lambda s: sub[f"acc_{s}"] >= 0.6  # noqa: E731
        ds_rows = {}
        for a, b in PAIRS:
            m = ok(a) & ok(b)
            if m.sum() == 0:
                continue
            diff = sub.loc[m, f"rt_{b}"] - sub.loc[m, f"rt_{a}"]
            ds_rows[f"{a}->{b}"] = {
                "n_shared": int(m.sum()),
                "mean_waste": round(float(diff.mean()), 1),
                "median_waste": round(float(diff.median()), 1),
                "total_waste": round(float(diff.sum()), 1),
                "pct_overhead": round(float((diff / sub.loc[m, f"rt_{a}"]).mean()), 4),
            }
            print(
                f"{ds:16} {a+'->'+b:10} {ds_rows[f'{a}->{b}']['n_shared']:>8} "
                f"{ds_rows[f'{a}->{b}']['mean_waste']:>10.1f} {ds_rows[f'{a}->{b}']['median_waste']:>12.1f} "
                f"{ds_rows[f'{a}->{b}']['total_waste']:>11.1f} {ds_rows[f'{a}->{b}']['pct_overhead']:>12.1%}"
            )
        out[ds] = ds_rows

    pooled = {}
    for a, b in PAIRS:
        n = sum(out[ds].get(f"{a}->{b}", {}).get("n_shared", 0) for ds in DATASETS)
        w = sum(out[ds].get(f"{a}->{b}", {}).get("total_waste", 0.0) for ds in DATASETS)
        pooled[f"{a}->{b}"] = {"n_shared": n, "total_waste": round(w, 1), "mean_waste_per_sample": round(w / n, 1) if n else None}
        print(f"pooled {a}->{b}: n={n} total={w:.0f} tokens (mean {w/n:.0f}/sample)")
    out["pooled"] = pooled

    write_json(args.output, out)
    print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
