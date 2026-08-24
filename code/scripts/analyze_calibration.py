#!/usr/bin/env python
"""M7 calibration analysis: accuracy, tokens, non-inferiority, flip rates.

Protocol: epsilon = 0.03, alpha = 0.05, paired McNemar-style analysis with
baseline = max reasoning setting.
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


def _mean(xs) -> float | None:
    vals = [x for x in xs if x is not None]
    return round(statistics.mean(vals), 1) if vals else None


def _mcnemar(s_ok: set, max_ok: set) -> dict:
    b = len(s_ok - max_ok)  # s correct, max wrong
    c = len(max_ok - s_ok)  # s wrong, max correct
    n_pairs = len(s_ok | max_ok)
    diff = (b - c) / n_pairs if n_pairs else 0.0
    se = math.sqrt(max(0.0, (b + c - ((b - c) ** 2) / n_pairs)) / (n_pairs ** 2)) if n_pairs else 0.0
    z = 1.645
    ci_lo = diff - z * se
    p = None
    if b + c > 0:
        # exact binomial two-sided p-value for discordant pairs
        from math import comb

        n_disc = b + c
        k = min(b, c)
        p = 2 * sum(comb(n_disc, i) * (0.5 ** n_disc) for i in range(k + 1))
        p = min(1.0, p)
    return {
        "s_only_correct": b,
        "max_only_correct": c,
        "difference": round(diff, 4),
        "ci_lower": round(ci_lo, 4),
        "mcnemar_p": round(p, 4) if p is not None else None,
        "non_inferior_at_epsilon_0_03": ci_lo >= -0.03,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Calibration analysis (M7).")
    ap.add_argument("--per-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "tables" / "calibration_analysis.json")
    return ap.parse_args()


def _stratum_map(dataset: str) -> dict[str, str]:
    pilot_cfg = load_yaml(ROOT / "code" / "configs" / "pilot_v1.yaml")
    path = Path(pilot_cfg["datasets"][dataset]["calibration"])
    samples = read_jsonl(path)
    return {s["id"]: str(s.get("_stratum")) for s in samples}


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.per_run)
    experiment = load_yaml(ROOT / "code" / "configs" / "experiment.yaml")
    epsilon = experiment["epsilon"]
    datasets = ["math500", "zebralogic_grid", "easy2hard_amc"]
    settings = ["low", "high", "max"]
    out: dict[str, object] = {"epsilon": epsilon, "n_rows": len(rows)}
    stratum_map = {ds: _stratum_map(ds) for ds in datasets}
    strata_by_ds = {}

    print(f"{'dataset':16} {'setting':7} {'n':>3} {'acc':>6} {'mean_rt':>9} {'median_rt':>9} {'cost':>7}")
    for ds in datasets:
        ds_rows = [r for r in rows if r["dataset"] == ds]
        ok_by_setting = {}
        stats_by_setting = {}
        for s in settings:
            v = [r for r in ds_rows if r["reasoning_setting"] == s]
            ok = {r["sample_id"] + f"#{r['repetition_id']}" for r in v if r["correct"] is True}
            ok_by_setting[s] = ok
            stats_by_setting[s] = {
                "n": len(v),
                "accuracy": round(sum(1 for r in v if r["correct"] is True) / len(v), 4) if v else None,
                "mean_reasoning_tokens": _mean([r["reasoning_tokens"] for r in v]),
                "median_reasoning_tokens": _mean([r["reasoning_tokens"] for r in v]) and round(
                    statistics.median([r["reasoning_tokens"] or 0 for r in v]), 1
                ),
                "cost_usd": round(sum(r["cost_usd"] or 0 for r in v), 4),
            }
            print(
                f"{ds:16} {s:7} {stats_by_setting[s]['n']:>3} {stats_by_setting[s]['accuracy'] or 0:>6.2f} "
                f"{stats_by_setting[s]['mean_reasoning_tokens'] or 0:>9.1f} "
                f"{stats_by_setting[s]['median_reasoning_tokens'] or 0:>9.1f} "
                f"{stats_by_setting[s]['cost_usd']:>7.4f}"
            )
        comparisons = {}
        for s in settings:
            if s == "max":
                continue
            comp = _mcnemar(ok_by_setting[s], ok_by_setting["max"])
            comp["token_savings_vs_max"] = (
                (stats_by_setting["max"]["mean_reasoning_tokens"] or 0)
                - (stats_by_setting[s]["mean_reasoning_tokens"] or 0)
            )
            comparisons[s] = comp
            print(
                f"  {ds} {s} vs max: diff={comp['difference']:.3f} ci_lo={comp['ci_lower']:.3f} "
                f"p={comp['mcnemar_p']} non_inferior={comp['non_inferior_at_epsilon_0_03']} "
                f"token_saving={comp['token_savings_vs_max']:.0f}"
            )
        out[ds] = {"settings": stats_by_setting, "vs_max": comparisons}

    print("\n== stratum x setting (n=18 per cell) ==")
    stratum_out: dict[str, object] = {}
    for ds in datasets:
        s2stratum = stratum_map[ds]
        cells = {}
        for r in rows:
            if r["dataset"] != ds:
                continue
            stratum = s2stratum.get(r["sample_id"], "?")
            cells.setdefault((stratum, r["reasoning_setting"]), []).append(r)
        table = {}
        for (stratum, setting), v in sorted(cells.items()):
            acc = sum(1 for x in v if x["correct"] is True) / len(v)
            mean_rt = statistics.mean(x["reasoning_tokens"] or 0 for x in v)
            table[f"{stratum}|{setting}"] = {"n": len(v), "accuracy": round(acc, 3), "mean_reasoning_tokens": round(mean_rt, 1)}
        strata_by_ds[ds] = sorted({s2stratum.get(r["sample_id"], "?") for r in rows if r["dataset"] == ds})
        stratum_out[ds] = table
        print(ds, "strata:", strata_by_ds[ds])
        for k, v in table.items():
            print(f"  {k:12} acc={v['accuracy']:.2f} mean_rt={v['mean_reasoning_tokens']:.0f}")
    out["stratum_analysis"] = stratum_out

    write_json(args.output, out)
    print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
