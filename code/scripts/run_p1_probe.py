#!/usr/bin/env python
"""Run the P1 probe (mechanism + prospective smoke) with per-item settings.

Each item runs only its probe settings (e.g., router setting + high) for
`repeats` repetitions. Results are cached by run key and appended to a per-run
JSONL. No condition runs unless its key is missing (resume-safe).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

from openai import OpenAI  # noqa: E402

from reasoning_efficiency.adapters.base import GenerationRequest  # noqa: E402
from reasoning_efficiency.adapters.openai_compatible import OpenAICompatibleAdapter  # noqa: E402
from reasoning_efficiency.eval import evaluate_answer  # noqa: E402
from reasoning_efficiency.io import load_yaml, read_jsonl, write_json  # noqa: E402
from reasoning_efficiency.prompt_builder import render_prompt  # noqa: E402


def _run_key(item: dict, setting: str, rep: int, prompt: str, model: str, version: str) -> str:
    payload = {
        "dataset": item["dataset"],
        "item_id": item["id"],
        "setting": setting,
        "rep": rep,
        "prompt": prompt,
        "model": model,
        "probe_version": version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:32]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "p1_probe.yaml")
    ap.add_argument("--scope", choices=["mechanism", "prospective", "all"], default="all")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-refresh", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)
    models_cfg = load_yaml(ROOT / "code" / "configs" / "models.yaml")
    prompts_cfg = load_yaml(ROOT / "code" / "configs" / "prompts.yaml")
    pricing_cfg = load_yaml(ROOT / "code" / "configs" / "pricing.yaml")
    provider = models_cfg["providers"][cfg["model"].split("-")[0]]
    model_cfg = models_cfg["models"][cfg["model"]]

    items = []
    if "item_files" in cfg:
        for f in cfg["item_files"]:
            for it in read_jsonl(Path(f)):
                if args.scope == "mechanism" and it["dataset"] != "MechanismProbe":
                    continue
                if args.scope == "prospective" and it["dataset"] == "MechanismProbe":
                    continue
                items.append(it)
    else:
        if args.scope in ("mechanism", "all"):
            items += read_jsonl(Path(cfg["mechanism_items"]))
        if args.scope in ("prospective", "all"):
            items += read_jsonl(Path(cfg["prospective_items"]))
    if args.limit:
        items = items[: args.limit]

    conditions = []
    for item in items:
        prompt_version = item["_prompt"]
        for setting in item["_probe_settings"]:
            for rep in range(cfg["repeats"]):
                conditions.append((item, setting, rep, prompt_version))

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    cache_dir = ROOT / "work" / "p1_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    per_run = ROOT / "work" / "metrics" / f"p1_probe_{run_id}.jsonl"
    lock = threading.Lock()

    if args.dry_run:
        print(f"[dry-run] {len(conditions)} probe conditions ({len(items)} items x {cfg['repeats']} reps)")
        return 0

    adapter = OpenAICompatibleAdapter(
        model_id=cfg["model"], provider_cfg=provider, model_cfg=model_cfg, wire_api=provider["wire_api"]
    )
    executed = skipped = failed = 0

    def process(cond) -> dict:
        nonlocal executed, skipped, failed
        item, setting, rep, prompt_version = cond
        key = _run_key(item, setting, rep, prompt_version, cfg["model"], cfg["schema_version"])
        cache_path = cache_dir / f"{key}.json"
        if cache_path.exists() and not args.force_refresh:
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            cached = True
        else:
            prompt = render_prompt(prompts_cfg["prompts"][prompt_version]["text"], item["question"])
            request = GenerationRequest(
                provider=provider["name"],
                model=cfg["model"],
                prompt=prompt,
                reasoning_control_type="effort",
                reasoning_setting=setting,
                temperature_requested=cfg["temperature"],
                seed_requested=cfg["seed"],
            )
            last = None
            for attempt in range(3):
                last = adapter.generate(request)
                if not last.error:
                    break
                time.sleep(1 + attempt * 2)
            price = pricing_cfg["prices_per_million_tokens"][cfg["model"]]
            if last is not None and last.error is None:
                cost = (
                    (last.input_tokens or 0) / 1e6 * price["input_cache_miss"]
                    + (last.output_tokens or 0) / 1e6 * price["output"]
                )
                last.cost_usd = round(cost, 8)
            record = {"key": key, "result": last.to_dict() if last else {}}
            if not record["result"].get("error"):
                cache_path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
            cached = False
        result = record["result"]
        correct = None
        if not result.get("error"):
            correct = evaluate_answer(item, result.get("response_text", ""))
        row = {
            "run_key": key,
            "dataset": item["dataset"],
            "item_id": item["id"],
            "stratum": item["stratum"],
            "setting": setting,
            "rep": rep,
            "prompt_version": prompt_version,
            "correct": correct,
            "reasoning_tokens": result.get("reasoning_tokens"),
            "output_tokens": result.get("output_tokens"),
            "input_tokens": result.get("input_tokens"),
            "latency_ms": result.get("latency_ms"),
            "cost_usd": result.get("cost_usd"),
            "error": result.get("error"),
            "cached": cached,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with lock:
            with open(per_run, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if cached:
                skipped += 1
            else:
                executed += 1
            if row["error"]:
                failed += 1
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process, c) for c in conditions]
        for f in as_completed(futures):
            f.result()

    summary = {
        "run_id": run_id,
        "planned": len(conditions),
        "executed": executed,
        "skipped_cached": skipped,
        "failed": failed,
        "per_run_path": str(per_run),
    }
    write_json(ROOT / "work" / "metrics" / f"p1_probe_{run_id}_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
