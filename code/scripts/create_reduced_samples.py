#!/usr/bin/env python
"""Reduced extension samples:

- GPQA-Diamond: first 30 samples (the 450 already-completed conditions).
- LiveCodeBench: 20 samples stratified by difficulty (7 easy / 7 medium / 6 hard), seed=42
  -> 20 x 3 settings x 5 repeats = 300 API calls.
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

from reasoning_efficiency.io import file_sha256, read_jsonl, write_json, write_jsonl  # noqa: E402


def _emit(path: Path, records: list[dict], seed: int, method: str, full_pool: int) -> None:
    write_jsonl(path, records)
    write_json(
        path.with_suffix(".manifest.json"),
        {
            "file": path.name,
            "records": len(records),
            "seed": seed,
            "method": method,
            "full_pool": full_pool,
            "strata": {s: sum(1 for r in records if r["_stratum"] == s) for s in sorted({r["_stratum"] for r in records})},
            "file_sha256": file_sha256(path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"wrote {path} ({len(records)} records)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=ROOT / "datasets" / "processed")
    ap.add_argument("--output", type=Path, default=ROOT / "datasets" / "pilot")
    ap.add_argument("--gpqa-n", type=int, default=30)
    ap.add_argument("--lcb-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    gpqa = read_jsonl(args.processed / "gpqa.jsonl")[: args.gpqa_n]
    for r in gpqa:
        r["_stratum"] = "1"
    _emit(args.output / "gpqa_diamond_pilot_v1.jsonl", gpqa, args.seed, f"first {args.gpqa_n}", len(read_jsonl(args.processed / "gpqa.jsonl")))

    lcb = read_jsonl(args.processed / "livecodebench.jsonl")
    rng = random.Random(args.seed)
    per = {"easy": 7, "medium": 7, "hard": 6}
    chosen: list[dict] = []
    for diff, n in per.items():
        pool = [r for r in lcb if r["metadata"].get("difficulty") == diff]
        if len(pool) < n:
            raise SystemExit(f"LCB {diff}: only {len(pool)} candidates (need {n})")
        idx = sorted(rng.sample(range(len(pool)), n))
        for i in idx:
            rec = dict(pool[i])
            rec["_stratum"] = diff
            chosen.append(rec)
    _emit(args.output / "livecodebench_pilot_v1.jsonl", chosen, args.seed, f"stratified {per}", len(lcb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

