from __future__ import annotations

import pytest

from reasoning_efficiency.adapters import MockAdapter, get_adapter
from reasoning_efficiency.adapters.base import GenerationRequest, ModelCapabilities
from reasoning_efficiency.adapters.openai_compatible import build_responses_params


def _request(setting: str = "high", temperature: float = 0.0, seed: int = 42) -> GenerationRequest:
    return GenerationRequest(
        provider="mock",
        model="mock-model",
        prompt="Question?",
        reasoning_control_type="effort",
        reasoning_setting=setting,
        temperature_requested=temperature,
        seed_requested=seed,
    )


def test_mock_adapter_deterministic_tokens():
    a = MockAdapter()
    low = a.generate(_request("low"))
    high = a.generate(_request("high"))
    assert low.reasoning_tokens == 100
    assert high.reasoning_tokens == 1000
    assert high.reasoning_tokens > low.reasoning_tokens
    assert high.requested_reasoning_setting == "high"
    assert high.effective_reasoning_setting == "high"


def test_mock_adapter_rejects_unknown_setting():
    a = MockAdapter()
    with pytest.raises(ValueError):
        a.generate(_request("medium"))


def test_mock_adapter_failure_setting():
    a = MockAdapter(fail_settings={"high"})
    result = a.generate(_request("high"))
    assert result.error == "mock failure"


def test_get_adapter_factory():
    assert isinstance(get_adapter("mock", model_id="m"), MockAdapter)
    with pytest.raises(KeyError):
        get_adapter("nope")


def test_capability_art_eligibility():
    caps = ModelCapabilities(
        supports_reasoning_effort=True,
        supported_reasoning_efforts=["low", "high"],
        supports_reasoning_budget=False,
        reports_reasoning_tokens=True,
    )
    assert caps.eligible_for_primary_art is True
    caps2 = ModelCapabilities(
        supports_reasoning_effort=True,
        supported_reasoning_efforts=["low", "high"],
        supports_reasoning_budget=False,
        reports_reasoning_tokens=False,
    )
    assert caps2.eligible_for_primary_art is False


def test_build_responses_params():
    caps = ModelCapabilities(
        supports_reasoning_effort=True,
        supported_reasoning_efforts=["low", "high", "max"],
        supports_reasoning_budget=False,
        supports_temperature=True,
        supports_seed=True,
    )
    params = build_responses_params(_request("max"), caps)
    assert params["model"] == "mock-model"
    assert params["reasoning"] == {"effort": "max"}
    assert params["temperature"] == 0.0
    assert params["seed"] == 42
