#!/usr/bin/env python
"""Create fixed stratified pilot / smoke / calibration samples (seed=42).

Strata:
  - math500: level 1..5 (20/stratum -> pilot 100; smoke 2/stratum; calibration 6/stratum)
  - zebralogic_grid: complexity quintile (20/stratum)
  - easy2hard_amc: difficulty quintile (20/stratum)

Smoke and calibration subsets are nested inside the pilot sample (first 2 / 6
per stratum in deterministic order).
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402

from reasoning_efficiency.io import file_sha256, load_yaml, read_jsonl, write_json, write_jsonl  # noqa: E402


def _stratified_sample(records: list[dict], strata: list[str], per_bin: int, seed: int, label: str) -> list[dict]:
    rng = random.Random(seed)
    selected: list[dict] = []
    for s in strata:
        pool = [r for r in records if r.get("_stratum") == s]
        if len(pool) < per_bin:
            raise SystemExit(f"{label}: stratum {s} has only {len(pool)} candidates (need {per_bin})")
        chosen = sorted(rng.sample(range(len(pool)), per_bin))
        for i in chosen:
            rec = dict(pool[i])
            rec["_stratum"] = s
            selected.append(rec)
    return selected


def _emit(name: str, records: list[dict], out_dir: Path, seed: int, full_count: int) -> None:
    path = out_dir / f"{name}.jsonl"
    write_jsonl(path, records)
    manifest = {
        "file": path.name,
        "records": len(records),
        "strata": {s: sum(1 for r in records if r["_stratum"] == s) for s in sorted({r["_stratum"] for r in records})},
        "seed": seed,
        "full_pool": full_count,
        "file_sha256": file_sha256(path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / f"{name}_manifest.json", manifest)
    print(f"wrote {path} ({len(records)} records)")


def _math500(processed: Path, out: Path, seed: int) -> None:
    records = read_jsonl(processed / "math500.jsonl")
    for r in records:
        r["_stratum"] = str(r["metadata"]["level"])
    strata = [str(i) for i in range(1, 6)]
    pilot = _stratified_sample(records, strata, 20, seed, "math500")
    by_stratum = {s: [r for r in pilot if r["_stratum"] == s] for s in strata}
    smoke = [r for s in strata for r in by_stratum[s][:2]]
    cal = [r for s in strata for r in by_stratum[s][:6]]
    _emit("math500_pilot_v1", pilot, out, seed, len(records))
    _emit("math500_smoke_v1", smoke, out, seed, len(pilot))
    _emit("math500_calibration_v1", cal, out, seed, len(pilot))


def _zebra_grid(processed: Path, out: Path, seed: int, complexity_csv: Path) -> None:
    records = read_jsonl(processed / "zebralogic.jsonl")
    grid = [r for r in records if r["metadata"].get("mode") == "grid"]
    comp = pd.read_csv(complexity_csv)
    quintile_map = dict(zip(comp["id"], comp["quintile"]))
    for r in grid:
        r["_stratum"] = str(int(quintile_map[r["id"]]))
    strata = [str(i) for i in range(1, 6)]
    pilot = _stratified_sample(grid, strata, 20, seed, "zebra_grid")
    by_stratum = {s: [r for r in pilot if r["_stratum"] == s] for s in strata}
    smoke = [r for s in strata for r in by_stratum[s][:2]]
    cal = [r for s in strata for r in by_stratum[s][:6]]
    _emit("zebralogic_grid_pilot_v1", pilot, out, seed, len(grid))
    _emit("zebralogic_grid_smoke_v1", smoke, out, seed, len(pilot))
    _emit("zebralogic_grid_calibration_v1", cal, out, seed, len(pilot))


def _easy2hard_amc(processed: Path, out: Path, seed: int) -> None:
    records = read_jsonl(processed / "easy2hard.jsonl")
    amc = [r for r in records if r["metadata"].get("subset") == "E2H-AMC"]
    diffs = [r["difficulty"] for r in amc]
    rank = pd.Series(diffs).rank(method="first", ascending=True).astype(int)
    n = len(amc)
    bin_size = -(-n // 5)
    quintile = ((rank - 1) // bin_size).clip(0, 4) + 1
    for r, q in zip(amc, quintile):
        r["_stratum"] = str(int(q))
    strata = [str(i) for i in range(1, 6)]
    pilot = _stratified_sample(amc, strata, 20, seed, "e2h_amc")
    by_stratum = {s: [r for r in pilot if r["_stratum"] == s] for s in strata}
    smoke = [r for s in strata for r in by_stratum[s][:2]]
    cal = [r for s in strata for r in by_stratum[s][:6]]
    _emit("easy2hard_amc_pilot_v1", pilot, out, seed, len(amc))
    _emit("easy2hard_amc_smoke_v1", smoke, out, seed, len(pilot))
    _emit("easy2hard_amc_calibration_v1", cal, out, seed, len(pilot))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Create fixed pilot/smoke/calibration samples.")
    ap.add_argument("--processed", type=Path, default=ROOT / "datasets" / "processed")
    ap.add_argument("--output", type=Path, default=ROOT / "datasets" / "pilot")
    ap.add_argument("--complexity-csv", type=Path, default=ROOT / "data" / "tables" / "zebra_complexity.csv")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _math500(args.processed, args.output, args.seed)
    _zebra_grid(args.processed, args.output, args.seed, args.complexity_csv)
    _easy2hard_amc(args.processed, args.output, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

