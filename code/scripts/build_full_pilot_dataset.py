#!/usr/bin/env python
"""Rebuild a consolidated full-pilot dataset from the result cache.

Cache is authoritative (one entry per run_key, no duplicates). For each cached
generation this script re-evaluates correctness deterministically and writes a
single JSONL for analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

from reasoning_efficiency.eval import evaluate_answer  # noqa: E402
from reasoning_efficiency.eval.zebra_grid import parse_model_output  # noqa: E402
from reasoning_efficiency.io import load_yaml, read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build consolidated full-pilot dataset from cache.")
    ap.add_argument("--cache", type=Path, default=ROOT / "work" / "cache")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "processed" / "full_pilot_v1.jsonl")
    ap.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "pilot_v1.yaml")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    pilot_cfg = load_yaml(args.config)
    sample_pool: dict[str, dict[str, dict]] = {}
    for dkey, dcfg in pilot_cfg["datasets"].items():
        for s in read_jsonl(Path(dcfg["pilot"])):
            sample_pool.setdefault(dkey, {})[s["id"]] = s

    # stored per-run evaluations (run_key -> {correct, parse_error}); avoids re-running
    # expensive code execution for LiveCodeBench.
    stored: dict[str, dict] = {}
    for per_run in (ROOT / "work" / "metrics").glob("*.jsonl"):
        for line in per_run.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            import json as _json

            row = _json.loads(line)
            if row.get("run_key"):
                stored[row["run_key"]] = {"correct": row.get("correct"), "parse_error": row.get("parse_error")}

    rows = []
    for path in sorted(args.cache.glob("*.json")):
        entry = json.loads(path.read_text(encoding="utf-8"))
        cond = entry["condition"]
        result = entry.get("result", {})
        dkey = cond["dataset"]
        sample = sample_pool.get(dkey, {}).get(cond["sample_id"])
        if sample is None:
            continue  # cached condition outside the reduced pilot sample (scope reduction)
        run_key = cond.get("run_key") or path.stem
        text = result.get("response_text", "")
        if run_key in stored and dkey == "livecodebench":
            correct = stored[run_key]["correct"]
            parse_error = stored[run_key]["parse_error"]
        else:
            correct = None
            parse_error = None
            if not result.get("error"):
                correct = evaluate_answer(sample, text)
                if dkey == "zebralogic_grid":
                    try:
                        parse_model_output(text)
                    except ValueError as e:
                        parse_error = str(e)
        rows.append(
            {
                "run_key": run_key,
                "dataset": dkey,
                "sample_id": cond["sample_id"],
                "stratum": str(sample.get("_stratum")),
                "model": cond["model"],
                "reasoning_setting": cond["reasoning_setting"],
                "repetition_id": cond["repetition_id"],
                "prompt_version": cond["prompt_version"],
                "correct": correct,
                "parse_error": parse_error,
                "error": result.get("error"),
                "reasoning_tokens": result.get("reasoning_tokens"),
                "output_tokens": result.get("output_tokens"),
                "input_tokens": result.get("input_tokens"),
                "total_tokens": result.get("total_tokens"),
                "latency_ms": result.get("latency_ms"),
                "cost_usd": result.get("cost_usd"),
                "raw_response_path": result.get("raw_response_path"),
            }
        )
    write_jsonl(args.output, rows)
    print(f"wrote {len(rows)} rows -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
