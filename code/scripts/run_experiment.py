#!/usr/bin/env python
"""Run experiment stages with cache/resume/dry-run/cost-estimation.

Usage:
    python scripts/run_experiment.py --stage smoke --adapter deepseek --models deepseek-v4-flash --run
    python scripts/run_experiment.py --stage calibration --adapter deepseek --estimate-cost
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

from reasoning_efficiency.adapters import MockAdapter, OpenAICompatibleAdapter  # noqa: E402
from reasoning_efficiency.adapters.base import ModelCapabilities  # noqa: E402
from reasoning_efficiency.io import load_yaml  # noqa: E402
from reasoning_efficiency.runner import ExperimentRunner  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run pilot experiment stages.")
    ap.add_argument("--config", type=Path, default=ROOT / "code" / "configs" / "pilot_v1.yaml")
    ap.add_argument("--stage", choices=["smoke", "calibration", "pilot"], default="smoke")
    ap.add_argument("--adapter", choices=["deepseek", "mock"], default=None)
    ap.add_argument("--models", default=None, help="comma-separated model ids")
    ap.add_argument("--datasets", default=None, help="comma-separated dataset keys")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-requests", type=int, default=None)
    ap.add_argument("--offset", type=int, default=0, help="skip first N conditions (parallel slicing)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--estimate-cost", action="store_true", help="sample N requests and extrapolate cost")
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--force-refresh", action="store_true")
    ap.add_argument("--run", action="store_true", help="execute after estimate (otherwise estimate-only)")
    ap.add_argument("--confirm", action="store_true", help="bypass cost/budget confirmation gate")
    ap.add_argument("--workers", type=int, default=1, help="parallel API workers (calibration/pilot)")
    return ap.parse_args()


def _make_adapter_factory(models_cfg: dict, provider_cfg: dict, adapter_name: str):
    def factory(model_id: str):
        if adapter_name == "mock":
            return MockAdapter(model_id=model_id)
        model_cfg = models_cfg["models"][model_id]
        caps = ModelCapabilities(
            supports_reasoning_effort=True,
            supported_reasoning_efforts=list(model_cfg["reasoning_settings"]),
            supports_reasoning_budget=False,
            reports_reasoning_tokens=True,
            exposes_reasoning_content=False,
            supports_temperature=True,
            supports_seed=True,
            reasoning_included_in_output_tokens=True,
        )
        return OpenAICompatibleAdapter(
            model_id=model_id,
            provider_cfg=provider_cfg,
            model_cfg=model_cfg,
            capabilities=caps,
            wire_api=provider_cfg.get("wire_api", "responses"),
        )

    return factory


def main() -> int:
    args = parse_args()
    pilot_cfg = load_yaml(args.config)
    experiment_cfg = load_yaml(ROOT / "code" / "configs" / "experiment.yaml")
    models_cfg = load_yaml(ROOT / "code" / "configs" / "models.yaml")
    prompts_cfg = load_yaml(ROOT / "code" / "configs" / "prompts.yaml")
    pricing_cfg = load_yaml(ROOT / "code" / "configs" / "pricing.yaml")
    provider_cfg = models_cfg["providers"][models_cfg["default_provider"]]
    adapter_name = args.adapter or pilot_cfg["run"]["adapter"]
    model_ids = [m.strip() for m in args.models.split(",")] if args.models else pilot_cfg["models"]
    dataset_keys = [d.strip() for d in args.datasets.split(",")] if args.datasets else None

    runner = ExperimentRunner(
        pilot_cfg=pilot_cfg,
        experiment_cfg=experiment_cfg,
        models_cfg=models_cfg,
        prompts_cfg=prompts_cfg,
        pricing_cfg=pricing_cfg,
        adapter_factory=_make_adapter_factory(models_cfg, provider_cfg, adapter_name),
    )

    if args.estimate_cost:
        est = runner.estimate_cost(
            args.stage, models=model_ids, datasets=dataset_keys, dry_run_samples=experiment_cfg["cost"]["dry_run_samples"]
        )
        print(f"[estimate] status={est.status} requests={est.requests} input={est.input_tokens} "
              f"output={est.output_tokens} reasoning={est.reasoning_tokens} cost_usd={est.cost_usd}")
        if est.status == "missing_prices":
            raise SystemExit("pricing.yaml has missing prices; refusing to proceed")
        if est.status == "errors":
            raise SystemExit(f"cost estimation failed: {est.note}")
        budget = experiment_cfg["cost"].get("calibration_budget_per_model")
        if args.stage == "calibration" and est.cost_usd is not None and est.cost_usd > budget and not args.confirm:
            raise SystemExit(f"estimated cost ${est.cost_usd:.2f} exceeds calibration budget ${budget}; use --confirm")
        if not args.run:
            return 0

    summary = runner.run(
        stage=args.stage,
        models=model_ids,
        datasets=dataset_keys,
        limit=args.limit,
        resume=args.resume,
        force_refresh=args.force_refresh,
        dry_run=args.dry_run,
        max_requests=args.max_requests,
        workers=args.workers,
        offset=args.offset,
    )
    print(f"per-run records: {summary.get('per_run_path')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
