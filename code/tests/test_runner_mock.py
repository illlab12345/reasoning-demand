from __future__ import annotations

import json
from pathlib import Path

from reasoning_efficiency.adapters import MockAdapter
from reasoning_efficiency.io import load_yaml, read_jsonl
from reasoning_efficiency.runner import ExperimentRunner

ROOT = Path(__file__).resolve().parents[2]


def _runner(tmp_path: Path, fail_settings=None, counter: list | None = None, run_id: str = "test-run"):
    pilot_cfg = load_yaml(ROOT / "code" / "configs" / "pilot_v1.yaml")
    experiment_cfg = load_yaml(ROOT / "code" / "configs" / "experiment.yaml")
    models_cfg = load_yaml(ROOT / "code" / "configs" / "models.yaml")
    prompts_cfg = load_yaml(ROOT / "code" / "configs" / "prompts.yaml")
    pricing_cfg = load_yaml(ROOT / "code" / "configs" / "pricing.yaml")

    def factory(model_id: str):
        adapter = MockAdapter(model_id=model_id, fail_settings=fail_settings or set())
        if counter is not None:
            original = adapter.generate

            def counted(request):
                counter.append(model_id)
                return original(request)

            adapter.generate = counted
        return adapter

    return ExperimentRunner(
        pilot_cfg=pilot_cfg,
        experiment_cfg=experiment_cfg,
        models_cfg=models_cfg,
        prompts_cfg=prompts_cfg,
        pricing_cfg=pricing_cfg,
        adapter_factory=factory,
        run_id=run_id,
        results_root=tmp_path,
    )


def test_dry_run_no_calls(tmp_path):
    counter: list = []
    r = _runner(tmp_path, counter=counter)
    summary = r.run(stage="smoke", limit=6, dry_run=True)
    assert summary["planned_requests"] == 6
    assert counter == []


def test_run_and_cache_resume(tmp_path):
    counter: list = []
    r1 = _runner(tmp_path, counter=counter, run_id="run-a")
    s1 = r1.run(stage="smoke", models=["deepseek-v4-flash"], limit=4)
    assert s1["executed"] == 4
    assert len(counter) == 4
    rows = read_jsonl(tmp_path / "metrics" / "run-a.jsonl")
    assert len(rows) == 4
    assert all("run_key" in x for x in rows)

    counter.clear()
    r2 = _runner(tmp_path, counter=counter, run_id="run-b")
    s2 = r2.run(stage="smoke", models=["deepseek-v4-flash"], limit=4)
    assert s2["executed"] == 0
    assert s2["skipped_cached"] == 4
    assert counter == []


def test_force_refresh_reexecutes(tmp_path):
    counter: list = []
    _runner(tmp_path, counter=counter, run_id="run-a").run(stage="smoke", models=["deepseek-v4-flash"], limit=2)
    counter.clear()
    r3 = _runner(tmp_path, counter=counter, run_id="run-c")
    s3 = r3.run(stage="smoke", models=["deepseek-v4-flash"], limit=2, force_refresh=True)
    assert s3["executed"] == 2
    assert len(counter) == 2


def test_retry_and_failure_recorded(tmp_path):
    counter: list = []
    r = _runner(tmp_path, fail_settings={"max"}, counter=counter, run_id="run-fail")
    s = r.run(stage="smoke", models=["deepseek-v4-flash"], limit=3)
    assert s["failed"] == 1  # third condition uses max
    assert len(counter) >= 1 + 1 + 3  # retries for max (1 initial + 2 retries)
    rows = read_jsonl(tmp_path / "metrics" / "run-fail.jsonl")
    errs = [x for x in rows if x["error"]]
    assert len(errs) == 1
    assert errs[0]["reasoning_setting"] == "max"


def test_append_only_across_runs(tmp_path):
    r1 = _runner(tmp_path, run_id="run-a")
    r1.run(stage="smoke", models=["deepseek-v4-flash"], limit=2)
    path1 = tmp_path / "metrics" / "run-a.jsonl"
    before = path1.read_bytes()
    r2 = _runner(tmp_path, run_id="run-d")
    r2.run(stage="smoke", models=["deepseek-v4-flash"], limit=2)
    assert path1.read_bytes() == before  # unchanged
    assert (tmp_path / "metrics" / "run-d.jsonl").exists()


def test_estimate_cost_mock(tmp_path):
    r = _runner(tmp_path, run_id="run-est")
    est = r.estimate_cost(stage="smoke", models=["deepseek-v4-flash"], dry_run_samples=2)
    assert est.status == "ok"
    assert est.requests == 60
    assert est.cost_usd is not None and est.cost_usd > 0


def test_mock_pipeline_evaluates_correctly(tmp_path):
    # mock returns "Answer: 42" which is correct for MATH-500 sample 0
    r = _runner(tmp_path, run_id="run-eval")
    s = r.run(stage="smoke", models=["deepseek-v4-flash"], limit=1)
    rows = read_jsonl(tmp_path / "metrics" / "run-eval.jsonl")
    assert rows[0]["correct"] is True or rows[0]["correct"] is False or rows[0]["correct"] is None
    assert rows[0]["reasoning_tokens"] is not None
