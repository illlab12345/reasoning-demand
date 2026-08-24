#!/usr/bin/env python
"""Validate WildEval/ZebraLogic mirror against allenai/ZebraLogicBench public.

Checks: ID overlap, normalized puzzle-text hash match, mc question/choices
match, grid size match. Writes results/tables/zebra_provenance.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pandas as pd  # noqa: E402

from reasoning_efficiency.io import write_json  # noqa: E402


def _norm(s) -> str:
    return " ".join(str(s).split()).casefold()


def _h(s) -> str:
    return hashlib.sha256(_norm(s).encode("utf-8")).hexdigest()


def _compare(mode: str, official_dir: Path, mirror_dir: Path) -> dict:
    rel = f"{mode}_mode/test-00000-of-00001.parquet"
    a = pd.read_parquet(official_dir / rel, engine="pyarrow")
    b = pd.read_parquet(mirror_dir / rel, engine="pyarrow")
    a_ids, b_ids = set(a["id"]), set(b["id"])
    a_map = {r["id"]: r for r in a.to_dict(orient="records")}
    b_map = {r["id"]: r for r in b.to_dict(orient="records")}
    common = a_ids & b_ids
    puzzle_match = sum(1 for i in common if _h(a_map[i]["puzzle"]) == _h(b_map[i]["puzzle"]))
    out = {
        "mode": mode,
        "official_rows": len(a),
        "mirror_rows": len(b),
        "id_overlap": len(common),
        "official_only": sorted(a_ids - b_ids)[:10],
        "mirror_only": sorted(b_ids - a_ids)[:10],
        "puzzle_text_hash_match": puzzle_match,
        "puzzle_text_hash_match_rate": round(puzzle_match / len(common), 6) if common else None,
    }
    if mode == "mc":
        q = sum(1 for i in common if _h(a_map[i]["question"]) == _h(b_map[i]["question"]))
        c = sum(1 for i in common if list(a_map[i]["choices"]) == list(b_map[i]["choices"]))
        out["question_hash_match_rate"] = round(q / len(common), 6)
        out["choices_exact_match_rate"] = round(c / len(common), 6)
    else:
        s = sum(1 for i in common if _h(a_map[i]["size"]) == _h(b_map[i]["size"]))
        out["size_match_rate"] = round(s / len(common), 6)
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validate ZebraLogic mirror provenance.")
    ap.add_argument("--official", type=Path, default=ROOT / "datasets" / "raw" / "zebralogic_allenai")
    ap.add_argument("--mirror", type=Path, default=ROOT / "datasets" / "raw" / "zebralogic")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "tables" / "zebra_provenance.json")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_repo": "allenai/ZebraLogicBench",
        "mirror_repo": "WildEval/ZebraLogic",
        "checks": [_compare("mc", args.official, args.mirror), _compare("grid", args.official, args.mirror)],
    }
    write_json(args.output, results)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

