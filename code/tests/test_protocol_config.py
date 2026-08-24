"""Protocol freeze tests: configs must be complete and mutually consistent."""

from __future__ import annotations

from pathlib import Path

from reasoning_efficiency.io import load_yaml

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str) -> dict:
    return load_yaml(ROOT / "code" / "configs" / name)


def test_config_files_exist():
    for name in ("experiment.yaml", "prompts.yaml", "models.yaml", "pricing.yaml"):
        assert (ROOT / "code" / "configs" / name).exists(), f"missing config: {name}"


def test_experiment_frozen_values():
    cfg = _load("experiment.yaml")
    assert cfg["status"] == "frozen"
    assert cfg["epsilon"] == 0.03
    assert cfg["alpha"] == 0.05
    assert cfg["repeats"]["calibration"] == 3
    assert cfg["repeats"]["full_pilot"] == 5
    assert cfg["temperature"] == 0.0
    assert cfg["seed"] == 42
    assert cfg["reasoning_baseline"] == "max"
    assert cfg["analysis_variable"] == "reasoning_tokens_observed"


def test_pilot_strata_consistent():
    cfg = _load("experiment.yaml")
    assert cfg["pilot_samples"] == {"math500": 100, "zebralogic_grid": 100, "easy2hard_amc": 100}
    for key, spec in cfg["strata"].items():
        assert spec["bins"] == 5
        assert spec["per_bin"] * spec["bins"] == cfg["pilot_samples"][key]


def test_prompts_render_and_are_neutral():
    prompts = _load("prompts.yaml")["prompts"]
    assert set(prompts) == {"math_v1", "zebra_grid_v1", "e2h_amc_v1", "aime_v1", "gpqa_v1", "livecodebench_v1"}
    for name, spec in prompts.items():
        text = spec["text"]
        assert "{question}" in text, f"{name}: missing {{question}} placeholder"
        lowered = text.lower()
        for banned in ("think step by step", "step-by-step", "请思考", "逐步思考"):
            assert banned not in lowered, f"{name}: prompt is not neutral ({banned!r})"


def test_prompt_versions_mapped():
    exp = _load("experiment.yaml")
    prompts = _load("prompts.yaml")["prompts"]
    mapping = exp["prompt_versions"]
    assert set(mapping) == {"math500", "zebralogic_grid", "easy2hard_amc", "aime", "gpqa_diamond", "livecodebench"}
    for dataset, prompt_key in mapping.items():
        assert prompt_key in prompts
        assert prompts[prompt_key]["version"] == "v1"


def test_models_config_consistent():
    models = _load("models.yaml")
    assert models["default_provider"] == "deepseek"
    assert "deepseek" in models["providers"]
    assert models["providers"]["deepseek"]["api_key_env"] == "DEEPSEEK_API_KEY"
    assert len(models["models"]) >= 1
    for mid, m in models["models"].items():
        assert m["provider"] in models["providers"]
        assert m["reasoning_settings"], f"{mid}: empty reasoning settings"
        assert m["max_reasoning_setting"] in m["reasoning_settings"]
        caps = m["capabilities"]
        for key in (
            "supports_reasoning_effort",
            "supports_reasoning_budget",
            "reports_reasoning_tokens",
            "exposes_reasoning_content",
            "supports_temperature",
            "supports_seed",
        ):
            assert key in caps


def test_pricing_config_covers_models():
    models = _load("models.yaml")["models"]
    prices = _load("pricing.yaml")["prices_per_million_tokens"]
    assert set(models) == set(prices), "every model must have a pricing entry"
    for mid, entry in prices.items():
        assert set(entry) >= {"input_cache_hit", "input_cache_miss", "output", "status"}
        assert entry["reasoning"] in ("same_as_output",) or isinstance(entry["reasoning"], (int, float))


def test_protocol_document_exists_and_frozen():
    doc = (ROOT / "docs" / "EXPERIMENT_PROTOCOL.md")
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    assert "0.03" in text
    assert "frozen" in text.lower() or "冻结" in text
