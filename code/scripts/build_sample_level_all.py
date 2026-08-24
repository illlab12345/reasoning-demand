#!/usr/bin/env python
"""Build sample-level flash data for ALL pilot benchmarks (core + extension).

Each row is one problem aggregated over its 5 repetitions: accuracy per
reasoning setting and mean reasoning tokens per setting. Stratum comes from the
pilot sample files.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

from reasoning_efficiency.io import load_yaml, read_jsonl  # noqa: E402

SETTINGS = ["low", "high", "max"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Sample-level flash data for all benchmarks.")
    ap.add_argument("--full-pilot", type=Path, default=ROOT / "data" / "processed" / "full_pilot_v1.jsonl")
    ap.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "pilot_v1.yaml")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "tables" / "flash_sample_level.csv")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    pilot_cfg = load_yaml(args.config)
    stratum: dict[str, dict[str, str]] = {}
    for dkey, dcfg in pilot_cfg["datasets"].items():
        p = Path(dcfg.get("pilot", ""))
        if not p.exists():
            continue
        for s in read_jsonl(p):
            stratum.setdefault(dkey, {})[s["id"]] = str(s.get("_stratum", "1"))

    agg: dict[tuple, list] = {}
    for line in open(args.full_pilot, encoding="utf-8"):
        r = json.loads(line)
        if r["model"] != "deepseek-v4-flash" or r["error"]:
            continue
        key = (r["dataset"], r["sample_id"], r["reasoning_setting"])
        agg.setdefault(key, []).append(r)

    by_sample: dict[tuple, dict] = {}
    for (ds, sid, s), v in agg.items():
        by_sample.setdefault((ds, sid), {})[s] = {
            "acc": sum(1 for x in v if x["correct"] is True) / len(v),
            "rt": statistics.mean(x["reasoning_tokens"] or 0 for x in v),
        }

    rows = []
    for (ds, sid), per in sorted(by_sample.items()):
        if set(per) != set(SETTINGS):
            continue  # only complete samples (all three settings present)
        rows.append(
            {
                "sample_id": sid,
                "dataset": ds,
                "stratum": stratum.get(ds, {}).get(sid, "1"),
                **{f"acc_{s}": round(per[s]["acc"], 3) for s in SETTINGS},
                **{f"rt_{s}": round(per[s]["rt"], 1) for s in SETTINGS},
            }
        )

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter

    print(f"wrote {len(rows)} samples -> {args.output}")
    print("by dataset:", dict(Counter(r["dataset"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

