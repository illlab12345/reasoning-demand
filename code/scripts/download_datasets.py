#!/usr/bin/env python
"""Download pinned pilot datasets from HuggingFace Hub into datasets/raw/<name>/.

Usage:
    python scripts/download_datasets.py [--datasets math500,zebralogic,easy2hard]
                                        [--dry-run] [--force] [--config PATH]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

from huggingface_hub import hf_hub_download  # noqa: E402

from reasoning_efficiency.io import load_yaml, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Download pinned pilot datasets from HuggingFace Hub.")
    ap.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "datasets.yaml")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset keys (default: all)")
    ap.add_argument("--dry-run", action="store_true", help="only print what would be downloaded")
    ap.add_argument("--force", action="store_true", help="re-download files even if they already exist")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    raw_root = ROOT / cfg["defaults"]["raw_dir"]
    keys = [k.strip() for k in args.datasets.split(",")] if args.datasets else list(cfg["datasets"])

    for key in keys:
        if key not in cfg["datasets"]:
            raise SystemExit(f"unknown dataset key: {key!r}")
        d = cfg["datasets"][key]
        target = raw_root / key
        target.mkdir(parents=True, exist_ok=True)

        files = list(d["files"]) + ["README.md"]
        for rel in files:
            dest = target / rel
            if dest.exists() and dest.stat().st_size > 0 and not args.force:
                print(f"[{key}] skip (exists): {rel}")
            else:
                action = "download" if args.force or not dest.exists() else f"re-download (size={dest.stat().st_size})"
                print(f"[{key}] {action}: {rel} @ {d['revision'][:12]}")

        if args.dry_run:
            continue

        manifest = {
            "dataset": d["name"],
            "key": key,
            "repo_id": d["repo_id"],
            "revision": d["revision"],
            "source_url": d["source_url"],
            "license": d.get("license"),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "files": [],
        }
        for rel in files:
            dest = target / rel
            if dest.exists() and dest.stat().st_size > 0 and not args.force:
                manifest["files"].append({"path": rel, "status": "exists", "size": dest.stat().st_size})
                continue
            print(f"[{key}] downloading {rel} ...")
            saved = hf_hub_download(
                repo_id=d["repo_id"],
                filename=rel,
                revision=d["revision"],
                repo_type="dataset",
                local_dir=str(target),
                force_download=args.force,
            )
            size = Path(saved).stat().st_size
            manifest["files"].append({"path": rel, "status": "downloaded", "size": size})
            print(f"[{key}] saved {saved} ({size} bytes)")

        manifest_path = target / "manifest.json"
        write_json(manifest_path, manifest)
        print(f"[{key}] manifest written: {manifest_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

