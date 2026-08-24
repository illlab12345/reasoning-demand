#!/usr/bin/env python
"""Probe model capabilities with minimal API calls and emit a capability matrix.

Usage:
    python scripts/probe_model_capabilities.py [--models deepseek-v4-flash,deepseek-v4-pro]
                                                [--adapter deepseek|mock] [--dry-run]
                                                [--write-config]

Each model is probed with a tiny prompt at the lowest and highest reasoning
setting (2 calls/model). The probe verifies: request succeeds, reasoning
setting is accepted, reasoning tokens are reported, reasoning content is
exposed, and token count increases with effort.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

from reasoning_efficiency.adapters import MockAdapter, OpenAICompatibleAdapter  # noqa: E402
from reasoning_efficiency.adapters.base import GenerationRequest, ModelCapabilities  # noqa: E402
from reasoning_efficiency.io import load_yaml, read_jsonl, write_json  # noqa: E402

def _hard_probe_prompt() -> str:
    """Use a real MATH-500 level-5 problem (probe only; not part of pilot analysis)."""
    records = read_jsonl(ROOT / "datasets" / "processed" / "math500.jsonl")
    hard = [r for r in records if r.get("difficulty") == 5]
    return hard[0]["question"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Probe model capabilities.")
    ap.add_argument("--models", default=None, help="comma-separated model ids (default: all in models.yaml)")
    ap.add_argument("--adapter", default="deepseek", choices=["deepseek", "mock"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--write-config", action="store_true", help="write probe results back into models.yaml")
    ap.add_argument("--output", type=Path, default=ROOT / "data" / "tables" / "model_capabilities.csv")
    return ap.parse_args()


def _probe_one(model_id: str, models_cfg: dict, provider_cfg: dict, adapter: str) -> dict:
    m = models_cfg[model_id]
    settings = list(m["reasoning_settings"])
    caps = ModelCapabilities(
        supports_reasoning_effort=True,
        supported_reasoning_efforts=settings,
        supports_reasoning_budget=False,
        supports_temperature=True,
        supports_seed=True,
    )
    if adapter == "mock":
        a = MockAdapter(model_id=model_id, reasoning_tokens_map={s: 100 + i * 100 for i, s in enumerate(settings)})
    else:
        a = OpenAICompatibleAdapter(model_id=model_id, provider_cfg=provider_cfg, model_cfg=m, capabilities=caps)

    results = {}
    probe_prompt = _hard_probe_prompt()
    for setting in settings:
        req = GenerationRequest(
            provider=provider_cfg["name"],
            model=model_id,
            prompt=probe_prompt,
            reasoning_control_type="effort",
            reasoning_setting=setting,
            temperature_requested=0.0,
            seed_requested=42,
        )
        results[setting] = a.generate(req)

    low, high = results[settings[0]], results[settings[-1]]
    reports_rt = all(r.reasoning_tokens is not None for r in results.values())
    tokens_increase = bool(
        reports_rt
        and high.reasoning_tokens is not None
        and low.reasoning_tokens is not None
        and high.reasoning_tokens > low.reasoning_tokens + 10
    )
    exposed = bool(
        any(getattr(r, "raw_metadata", {}).get("has_reasoning_item") for r in results.values())
    )
    debug_dir = ROOT / "work" / "probe_raw"
    debug_dir.mkdir(parents=True, exist_ok=True)
    for setting, r in results.items():
        write_json(
            debug_dir / f"{model_id}_{setting}.json",
            {
                "model": model_id,
                "setting": setting,
                "error": r.error,
                "model_version": r.model_version,
                "finish_reason": r.finish_reason,
                "usage": r.raw_usage,
                "raw_metadata": r.raw_metadata,
                "response_text": r.response_text[:300],
                "reasoning_tokens": r.reasoning_tokens,
                "output_tokens": r.output_tokens,
                "input_tokens": r.input_tokens,
            },
        )
    temperature_ok = all(r.temperature_effective == 0.0 for r in results.values())
    seed_ok = all(r.seed_effective == 42 for r in results.values())
    return {
        "model": model_id,
        "settings_probed": settings,
        "low_setting": settings[0],
        "high_setting": settings[-1],
        "low_error": low.error,
        "high_error": high.error,
        "all_settings": {s: {"error": r.error, "reasoning_tokens": r.reasoning_tokens, "output_tokens": r.output_tokens} for s, r in results.items()},
        "low_reasoning_tokens": low.reasoning_tokens,
        "high_reasoning_tokens": high.reasoning_tokens,
        "low_output_tokens": low.output_tokens,
        "high_output_tokens": high.output_tokens,
        "reports_reasoning_tokens": reports_rt and not low.error and not high.error,
        "reasoning_tokens_increase_with_effort": tokens_increase,
        "exposes_reasoning_content": exposed,
        "model_version": low.model_version or high.model_version,
        "eligible_for_primary_art": bool(reports_rt and tokens_increase and not low.error and not high.error),
        "supports_temperature": temperature_ok,
        "supports_seed": seed_ok,
        "probed_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    cfg = load_yaml(ROOT / "code" / "configs" / "models.yaml")
    models_cfg = cfg["models"]
    provider_cfg = cfg["providers"][cfg["default_provider"]]
    model_ids = [x.strip() for x in args.models.split(",")] if args.models else list(models_cfg)

    rows = []
    for mid in model_ids:
        if mid not in models_cfg:
            raise SystemExit(f"unknown model: {mid}")
        print(f"[probe] {mid} ({args.adapter}) settings={models_cfg[mid]['reasoning_settings']}")
        if args.dry_run:
            print(f"[probe] dry-run: would call {args.adapter} for {mid}")
            continue
        row = _probe_one(mid, models_cfg, provider_cfg, args.adapter)
        rows.append(row)
        print(f"  reports_reasoning_tokens={row['reports_reasoning_tokens']} "
              f"low={row['low_reasoning_tokens']} high={row['high_reasoning_tokens']} "
              f"eligible={row['eligible_for_primary_art']}")

    if not rows:
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved -> {args.output}")

    if args.write_config and args.adapter == "deepseek":
        probe_map = {r["model"]: r for r in rows}
        for mid in model_ids:
            if mid in models_cfg and mid in probe_map:
                p = probe_map[mid]
                caps = models_cfg[mid]["capabilities"]
                caps["reports_reasoning_tokens"] = bool(p["reports_reasoning_tokens"])
                caps["exposes_reasoning_content"] = bool(p["exposes_reasoning_content"])
                caps["supports_temperature"] = not p["low_error"]
                caps["supports_seed"] = not p["low_error"]
                models_cfg[mid]["probe"] = p
        write_json(ROOT / "code" / "configs" / "models.yaml", cfg)
        print("updated configs/models.yaml with probe results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
