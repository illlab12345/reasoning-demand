#!/usr/bin/env python
"""Convert raw pilot datasets into unified JSONL schema (datasets/processed/).

Usage:
    python scripts/prepare_datasets.py [--datasets math500,zebralogic,easy2hard]
                                       [--pilot-limit N] [--pilot-seed S]
                                       [--overwrite]
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

from reasoning_efficiency.io import file_sha256, load_yaml, read_json, write_json, write_jsonl  # noqa: E402
from reasoning_efficiency.loaders import get_loader  # noqa: E402
from reasoning_efficiency.schema import SCHEMA_VERSION, validate_records  # noqa: E402


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert raw datasets to unified JSONL schema.")
    ap.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "datasets.yaml")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset keys (default: all)")
    ap.add_argument("--pilot-limit", type=int, default=None, help="write a pilot subset of N records per dataset")
    ap.add_argument("--pilot-seed", type=int, default=None, help="seed for deterministic pilot sampling")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing processed files")
    return ap.parse_args()


def select_pilot(records: list[dict], limit: int, seed: int | None) -> list[dict]:
    if limit >= len(records):
        return list(records)
    if seed is None:
        return list(records[:limit])
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), limit)
    return [records[i] for i in sorted(indices)]


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    defaults = cfg["defaults"]
    raw_root = ROOT / defaults["raw_dir"]
    processed_dir = ROOT / defaults["processed_dir"]
    pilot_dir = ROOT / defaults["pilot_dir"]
    keys = [k.strip() for k in args.datasets.split(",")] if args.datasets else list(cfg["datasets"])

    for key in keys:
        if key not in cfg["datasets"]:
            raise SystemExit(f"unknown dataset key: {key!r}")
        d = cfg["datasets"][key]
        loader = get_loader(d["loader"])
        loaded = loader(raw_root / key, d)

        validation = validate_records(loaded.records, dataset=d["name"])
        if validation["n_errors"]:
            raise SystemExit(
                f"[{key}] schema validation failed: {validation['n_errors']} errors; "
                f"first: {validation['errors'][:10]}"
            )
        if not validation["unique_ids"]:
            raise SystemExit(f"[{key}] duplicate ids: {validation['duplicate_ids']}")

        out_path = processed_dir / f"{key}.jsonl"
        if out_path.exists() and not args.overwrite:
            raise SystemExit(f"[{key}] {out_path} already exists (use --overwrite to regenerate)")
        write_jsonl(out_path, loaded.records, overwrite=args.overwrite)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "dataset": d["name"],
            "repo_id": d["repo_id"],
            "revision": d["revision"],
            "records": len(loaded.records),
            "row_counts": loaded.row_counts,
            "raw_columns": loaded.raw_columns,
            "skipped_rows": loaded.skipped_rows,
            "file_sha256": file_sha256(out_path),
            "prepared_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit(),
        }
        manifest_path = processed_dir / f"{key}_manifest.json"
        write_json(manifest_path, manifest)
        print(f"[{key}] wrote {len(loaded.records)} records -> {out_path}")

        if args.pilot_limit:
            pilot_records = select_pilot(loaded.records, args.pilot_limit, args.pilot_seed)
            pilot_path = pilot_dir / f"{key}.jsonl"
            write_jsonl(pilot_path, pilot_records, overwrite=args.overwrite)
            pilot_manifest = {
                "dataset": d["name"],
                "limit": len(pilot_records),
                "method": "seeded random sample" if args.pilot_seed is not None else "first N",
                "seed": args.pilot_seed,
                "full_records": len(loaded.records),
                "file_sha256": file_sha256(pilot_path),
                "prepared_at": datetime.now(timezone.utc).isoformat(),
            }
            write_json(pilot_dir / f"{key}_manifest.json", pilot_manifest)
            print(f"[{key}] pilot subset -> {pilot_path} ({len(pilot_records)} records)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
