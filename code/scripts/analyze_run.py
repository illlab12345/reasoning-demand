#!/usr/bin/env python
"""Analyze a completed run's per-run records.

Usage:
    python scripts/analyze_run.py --per-run results/metrics/<run_id>.jsonl
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

from reasoning_efficiency.io import read_jsonl, write_json  # noqa: E402


def _mean(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.mean(vals), 2) if vals else None


def _median(values: list) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.median(vals), 2) if vals else None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Analyze a run.")
    ap.add_argument("--per-run", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.per_run)
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["dataset"], r["model"], r["reasoning_setting"]), []).append(r)

    table: dict[str, object] = {}
    for key in sorted(groups):
        v = groups[key]
        acc = sum(1 for r in v if r["correct"] is True) / len(v)
        table["|".join(key)] = {
            "n": len(v),
            "accuracy": round(acc, 4),
            "mean_reasoning_tokens": _mean([r["reasoning_tokens"] for r in v]),
            "median_reasoning_tokens": _median([r["reasoning_tokens"] for r in v]),
            "mean_latency_ms": _mean([r["latency_ms"] for r in v]),
            "errors": sum(1 for r in v if r["error"]),
            "cost_usd": round(sum(r["cost_usd"] or 0 for r in v), 4),
        }

    print(f"{'dataset | model | setting':44} {'n':>3} {'acc':>6} {'mean_rt':>9} {'med_rt':>8} {'lat_ms':>8} {'cost':>7}")
    for key, s in table.items():
        print(
            f"{key:44} {s['n']:>3} {s['accuracy']:>6.2f} {s['mean_reasoning_tokens'] or 0:>9.0f} "
            f"{s['median_reasoning_tokens'] or 0:>8.0f} {s['mean_latency_ms'] or 0:>8.0f} {s['cost_usd']:>7.4f}"
        )

    # grid parse-failure rate (zebra grid only): re-evaluate raw responses
    grid_rows = [r for r in rows if r["dataset"] == "zebralogic_grid"]
    parse_fail = 0
    for r in grid_rows:
        raw_path = r.get("raw_response_path")
        if not raw_path or not Path(raw_path).exists():
            continue
        raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        text = raw.get("result", {}).get("response_text", "")
        try:
            from reasoning_efficiency.eval.zebra_grid import parse_model_output

            parse_model_output(text)
        except ValueError:
            parse_fail += 1
    grid_stats = {
        "n": len(grid_rows),
        "parse_failures": parse_fail,
        "parse_failure_rate": round(parse_fail / len(grid_rows), 4) if grid_rows else None,
    }
    table["_grid_parse"] = grid_stats
    print(f"grid parse failures: {parse_fail}/{len(grid_rows)}")

    out_path = args.output or args.per_run.with_suffix(".analysis.json")
    write_json(out_path, {"run": args.per_run.name, "groups": table, "total_cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 4)})
    print(f"saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

