#!/usr/bin/env python
"""Expand GPQA-Diamond and LiveCodeBench pilot samples to 100 problems each.

Keeps the previously sampled problems (so cached conditions remain valid) and
fills the remainder deterministically with seed 42:
- GPQA-Diamond: keep first 30, add 70 from the remaining 168.
- LiveCodeBench: keep 7/7/6 (easy/medium/hard), add to 35/43/22 stratified.
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
    ap.add_argument("--pilot", type=Path, default=ROOT / "datasets" / "pilot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # GPQA: keep existing 30 (first 30 of processed), add 70 from the rest
    gpqa = read_jsonl(args.processed / "gpqa.jsonl")
    existing_gpqa = {r["id"] for r in read_jsonl(args.pilot / "gpqa_diamond_pilot_v1.jsonl")}
    keep = [r for r in gpqa if r["id"] in existing_gpqa]
    pool = [r for r in gpqa if r["id"] not in existing_gpqa]
    add = [pool[i] for i in sorted(rng.sample(range(len(pool)), 100 - len(keep)))]
    gpqa_out = keep + add
    for r in gpqa_out:
        r["_stratum"] = "1"
    _emit(args.pilot / "gpqa_diamond_pilot_v1.jsonl", gpqa_out, args.seed, "keep 30 + random 70 (seed 42)", len(gpqa))

    # LCB: keep existing 7/7/6, fill to 35/43/22
    lcb = read_jsonl(args.processed / "livecodebench.jsonl")
    existing_lcb = {r["id"] for r in read_jsonl(args.pilot / "livecodebench_pilot_v1.jsonl")}
    targets = {"easy": 35, "medium": 43, "hard": 22}
    lcb_out = []
    for diff, target in targets.items():
        same = [r for r in lcb if r["id"] in existing_lcb and r["metadata"].get("difficulty") == diff]
        pool = [r for r in lcb if r["id"] not in existing_lcb and r["metadata"].get("difficulty") == diff]
        need = target - len(same)
        if need < 0 or len(pool) < need:
            raise SystemExit(f"LCB {diff}: keep={len(same)} need={need} pool={len(pool)}")
        chosen = same + [pool[i] for i in sorted(rng.sample(range(len(pool)), need))]
        for r in chosen:
            r["_stratum"] = diff
        lcb_out.extend(chosen)
    _emit(args.pilot / "livecodebench_pilot_v1.jsonl", lcb_out, args.seed, f"keep 7/7/6 -> 35/43/22 (seed 42)", len(lcb))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

