#!/usr/bin/env python
"""Create fixed pilot samples for the three extension benchmarks.

- AIME: all 30 problems (small benchmark; no sampling).
- GPQA-Diamond: all 198 problems (full benchmark).
- LiveCodeBench: 60 problems stratified by difficulty (20 easy / 20 medium / 20 hard), seed=42.
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


def _emit(path: Path, records: list[dict], seed: int, full_pool: int, method: str) -> None:
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
    ap = argparse.ArgumentParser(description="Extension pilot samples.")
    ap.add_argument("--processed", type=Path, default=ROOT / "datasets" / "processed")
    ap.add_argument("--output", type=Path, default=ROOT / "datasets" / "pilot")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lcb-n", type=int, default=60)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    aime = read_jsonl(args.processed / "aime.jsonl")
    for r in aime:
        r["_stratum"] = "1"
    _emit(args.output / "aime_pilot_v1.jsonl", aime, args.seed, len(aime), "all")

    gpqa = read_jsonl(args.processed / "gpqa.jsonl")
    for r in gpqa:
        r["_stratum"] = "1"
    _emit(args.output / "gpqa_diamond_pilot_v1.jsonl", gpqa, args.seed, len(gpqa), "all")

    lcb = read_jsonl(args.processed / "livecodebench.jsonl")
    rng = random.Random(args.seed)
    per = args.lcb_n // 3
    chosen: list[dict] = []
    for diff in ("easy", "medium", "hard"):
        pool = [r for r in lcb if r["metadata"].get("difficulty") == diff]
        if len(pool) < per:
            raise SystemExit(f"LCB {diff}: only {len(pool)} candidates (need {per})")
        idx = sorted(rng.sample(range(len(pool)), per))
        for i in idx:
            rec = dict(pool[i])
            rec["_stratum"] = diff
            chosen.append(rec)
    _emit(args.output / "livecodebench_pilot_v1.jsonl", chosen, args.seed, len(lcb), f"stratified {per}x3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

