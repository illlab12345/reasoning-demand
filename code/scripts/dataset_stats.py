#!/usr/bin/env python
"""Compute and save statistics for processed pilot datasets."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

from reasoning_efficiency.io import load_yaml, read_jsonl, write_json  # noqa: E402
from reasoning_efficiency.stats import compute_statistics, format_statistics  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute dataset statistics.")
    ap.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "datasets.yaml")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset keys (default: all)")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "tables" / "dataset_stats.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    processed_dir = ROOT / cfg["defaults"]["processed_dir"]
    keys = [k.strip() for k in args.datasets.split(",")] if args.datasets else list(cfg["datasets"])

    all_stats: dict[str, object] = {}
    for key in keys:
        path = processed_dir / f"{key}.jsonl"
        if not path.exists():
            raise SystemExit(f"processed file not found: {path} (run prepare_datasets.py first)")
        records = read_jsonl(path)
        stats = compute_statistics(records)
        all_stats[key] = stats
        print(f"===== {key} =====")
        print(format_statistics(stats))
        print()

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "datasets": all_stats,
    }
    write_json(args.output, out)
    print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

