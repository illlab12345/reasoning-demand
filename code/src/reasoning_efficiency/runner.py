"""Experiment runner: inference -> parse -> evaluate -> append-only persistence."""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .adapters.base import BaseAdapter, GenerationRequest
from .cost import CostEstimate, estimate, extrapolate
from .eval import evaluate_answer
from .io import read_jsonl, write_json, write_jsonl
from .parsing import parse_answer_for_record
from .prompt_builder import render_prompt


@dataclass(frozen=True)
class ExperimentCondition:
    dataset_key: str
    sample: dict[str, Any]
    model_id: str
    reasoning_setting: str
    repetition_id: int
    prompt_version: str
    generation_config: dict[str, Any]

    def run_key(self) -> str:
        payload = {
            "dataset": self.dataset_key,
            "sample_id": self.sample["id"],
            "model": self.model_id,
            "prompt_version": self.prompt_version,
            "reasoning_setting": self.reasoning_setting,
            "repetition_id": self.repetition_id,
            "generation_config": self.generation_config,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:32]


class ExperimentRunner:
    def __init__(
        self,
        pilot_cfg: dict[str, Any],
        experiment_cfg: dict[str, Any],
        models_cfg: dict[str, Any],
        prompts_cfg: dict[str, Any],
        pricing_cfg: dict[str, Any],
        adapter_factory: Callable[[str], BaseAdapter],
        run_id: str | None = None,
        results_root: Path | None = None,
    ):
        self.pilot_cfg = pilot_cfg
        self.experiment_cfg = experiment_cfg
        self.models_cfg = models_cfg
        self.prompts_cfg = prompts_cfg
        self.pricing_cfg = pricing_cfg
        self.adapter_factory = adapter_factory
        self.run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.results_root = results_root or Path(pilot_cfg["run"]["results_root"])
        self.raw_dir = self.results_root / "raw_generations" / self.run_id
        self.metrics_dir = self.results_root / "metrics"
        self.cache_dir = self.results_root / "cache"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self._per_run_path = self.metrics_dir / f"{self.run_id}.jsonl"
        self._io_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._adapter_cache: dict[str, BaseAdapter] = {}

    def conditions(
        self, stage: str, models: list[str] | None = None, datasets: list[str] | None = None
    ) -> list[ExperimentCondition]:
        out: list[ExperimentCondition] = []
        repeats = self.pilot_cfg.get("stages", {}).get(stage, {}).get("repeats") or self.experiment_cfg["repeats"]["calibration"]
        gen_cfg = {
            "temperature": self.experiment_cfg["temperature"],
            "seed": self.experiment_cfg["seed"],
        }
        model_ids = models or self.pilot_cfg["models"]
        for dkey, dcfg in self.pilot_cfg["datasets"].items():
            if datasets is not None and dkey not in datasets:
                continue
            if stage not in dcfg:
                continue  # dataset not part of this stage (e.g. extension datasets have pilot only)
            samples_path = Path(dcfg[stage])
            if not samples_path.exists():
                raise FileNotFoundError(f"stage sample file missing: {samples_path}")
            samples = read_jsonl(samples_path)
            prompt_version = dcfg["prompt"]
            for sample in samples:
                for mid in model_ids:
                    settings = self.models_cfg["models"][mid]["reasoning_settings"]
                    stage_settings = self.pilot_cfg.get("stages", {}).get(stage, {}).get("settings")
                    if stage_settings:
                        settings = [s for s in settings if s in stage_settings]
                    for setting in settings:
                        for rep in range(repeats):
                            out.append(
                                ExperimentCondition(
                                    dataset_key=dkey,
                                    sample=sample,
                                    model_id=mid,
                                    reasoning_setting=setting,
                                    repetition_id=rep,
                                    prompt_version=prompt_version,
                                    generation_config=gen_cfg,
                                )
                            )
        return out

    def _adapter(self, model_id: str) -> BaseAdapter:
        if model_id not in self._adapter_cache:
            self._adapter_cache[model_id] = self.adapter_factory(model_id)
        return self._adapter_cache[model_id]

    def _generate(self, condition: ExperimentCondition) -> tuple[dict[str, Any], float | None]:
        adapter = self._adapter(condition.model_id)
        prompt = render_prompt(
            self.prompts_cfg["prompts"][condition.prompt_version]["text"],
            condition.sample["question"],
        )
        request = GenerationRequest(
            provider=self.models_cfg["providers"][self.models_cfg["default_provider"]]["name"],
            model=condition.model_id,
            prompt=prompt,
            reasoning_control_type="effort",
            reasoning_setting=condition.reasoning_setting,
            temperature_requested=condition.generation_config.get("temperature"),
            seed_requested=condition.generation_config.get("seed"),
        )
        max_retries = self.experiment_cfg["max_retries"]
        backoffs = self.experiment_cfg["retry_backoff_seconds"]
        attempts = 0
        result = None
        while attempts <= max_retries:
            result = adapter.generate(request)
            if not result.error:
                break
            if attempts < max_retries:
                delay = backoffs[min(attempts, len(backoffs) - 1)]
                time.sleep(delay)
            attempts += 1
        cost = None
        if result is not None:
            price = self.pricing_cfg["prices_per_million_tokens"].get(condition.model_id, {})
            reasoning_included = getattr(adapter.capabilities, "reasoning_included_in_output_tokens", None)
            if price:
                from .cost import request_cost_usd

                cost = request_cost_usd(result, price, reasoning_included)
            result.cost_usd = cost if cost is not None else 0.0
        record = {
            "condition": {
                "dataset": condition.dataset_key,
                "sample_id": condition.sample["id"],
                "model": condition.model_id,
                "reasoning_setting": condition.reasoning_setting,
                "repetition_id": condition.repetition_id,
                "prompt_version": condition.prompt_version,
                "generation_config": condition.generation_config,
            },
            "run_key": condition.run_key(),
            "result": result.to_dict() if result is not None else {},
            "attempts": attempts,
        }
        return record, cost

    def run(
        self,
        stage: str,
        models: list[str] | None = None,
        datasets: list[str] | None = None,
        limit: int | None = None,
        resume: bool = True,
        force_refresh: bool = False,
        dry_run: bool = False,
        max_requests: int | None = None,
        workers: int = 1,
        offset: int = 0,
    ) -> dict[str, Any]:
        conditions = self.conditions(stage, models=models, datasets=datasets)
        if offset:
            conditions = conditions[offset:]
        if limit is not None:
            conditions = conditions[:limit]
        if max_requests is not None:
            conditions = conditions[:max_requests]
        summary = {
            "run_id": self.run_id,
            "stage": stage,
            "planned_requests": len(conditions),
            "per_run_path": str(self._per_run_path),
        }
        if dry_run:
            print(f"[dry-run] {self.run_id}: would execute {len(conditions)} requests (stage={stage})")
            return summary

        executed = skipped = failed = 0
        progress = {"done": 0}

        def process(cond: ExperimentCondition) -> tuple[dict[str, Any], bool, bool]:
            nonlocal executed, skipped, failed
            key = cond.run_key()
            cache_path = self.cache_dir / f"{key}.json"
            if resume and cache_path.exists() and not force_refresh:
                record = json.loads(cache_path.read_text(encoding="utf-8"))
                if record.get("result", {}).get("error"):
                    # never resume from a failed attempt; re-execute
                    record, _ = self._generate(cond)
                    record["condition"]["cached"] = False
                    write_json(cache_path, record)
                    is_cached = False
                else:
                    record["condition"]["cached"] = True
                    is_cached = True
            else:
                record, _ = self._generate(cond)
                record["condition"]["cached"] = False
                if not record.get("result", {}).get("error"):
                    write_json(cache_path, record)
                is_cached = False
            raw_path = self.raw_dir / f"{key}.json"
            if not raw_path.exists():
                write_json(raw_path, record)
            record["result"]["raw_response_path"] = str(raw_path)
            self._append_per_run(cond, record)
            has_error = bool(record.get("result", {}).get("error"))
            with self._progress_lock:
                progress["done"] += 1
                if not is_cached:
                    executed += 1
                else:
                    skipped += 1
                if has_error:
                    failed += 1
                if progress["done"] % 25 == 0:
                    print(f"[{self.run_id}] {progress['done']}/{len(conditions)}")
            return record, is_cached, has_error

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(process, cond) for cond in conditions]
                for future in as_completed(futures):
                    future.result()
        else:
            for cond in conditions:
                process(cond)
        summary.update(
            {
                "executed": executed,
                "skipped_cached": skipped,
                "failed": failed,
                "completed": progress["done"],
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(self.metrics_dir / f"{self.run_id}_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return summary

    def _append_per_run(self, condition: ExperimentCondition, record: dict[str, Any]) -> None:
        result = record.get("result", {})
        correct = None
        if not result.get("error") and result.get("response_text"):
            correct = evaluate_answer(condition.sample, result["response_text"])
        parsed = parse_answer_for_record(condition.sample, result.get("response_text", ""))
        row = {
            "run_key": condition.run_key(),
            "dataset": condition.dataset_key,
            "sample_id": condition.sample["id"],
            "model": condition.model_id,
            "reasoning_setting": condition.reasoning_setting,
            "repetition_id": condition.repetition_id,
            "prompt_version": condition.prompt_version,
            "cached": record.get("condition", {}).get("cached", False),
            "correct": correct,
            "parsed_answer": parsed,
            "reasoning_tokens": result.get("reasoning_tokens"),
            "output_tokens": result.get("output_tokens"),
            "input_tokens": result.get("input_tokens"),
            "total_tokens": result.get("total_tokens"),
            "latency_ms": result.get("latency_ms"),
            "cost_usd": result.get("cost_usd"),
            "raw_response_path": result.get("raw_response_path"),
            "error": result.get("error"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._io_lock:
            with open(self._per_run_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def estimate_cost(
        self, stage: str, models: list[str] | None = None, datasets: list[str] | None = None, dry_run_samples: int = 10
    ) -> CostEstimate:
        conditions = self.conditions(stage, models=models, datasets=datasets)
        # stratified sample: round-robin across (dataset, setting) groups so the
        # extrapolation is not dominated by the cheapest first conditions
        groups: dict[tuple[str, str], list[ExperimentCondition]] = {}
        for c in conditions:
            groups.setdefault((c.dataset_key, c.reasoning_setting), []).append(c)
        sample: list[ExperimentCondition] = []
        keys = sorted(groups)
        idx = 0
        while len(sample) < min(dry_run_samples, len(conditions)):
            key = keys[idx % len(keys)]
            group = groups[key]
            if group:
                sample.append(group.pop(0))
            idx += 1
            if sum(len(g) for g in groups.values()) == 0:
                break
        results = []
        for cond in sample:
            record, _ = self._generate(cond)
            results.append(record["result"])
        est = estimate(
            results,
            self.pricing_cfg["prices_per_million_tokens"],
            reasoning_included=False,
        )
        return extrapolate(est, len(sample), len(conditions))
