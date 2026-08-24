"""Deterministic mock adapter for offline pipeline tests."""

from __future__ import annotations

import time

from .base import GenerationRequest, GenerationResult, ModelCapabilities


class MockAdapter:
    def __init__(
        self,
        model_id: str = "mock-model",
        reasoning_tokens_map: dict[str, int] | None = None,
        response_text: str = "Answer: 42",
        fail_settings: set[str] | None = None,
    ):
        self.model_id = model_id
        self.reasoning_tokens_map = reasoning_tokens_map or {"low": 100, "high": 1000, "max": 5000}
        self.response_text = response_text
        self.fail_settings = fail_settings or set()
        self.capabilities = ModelCapabilities(
            supports_reasoning_effort=True,
            supported_reasoning_efforts=list(self.reasoning_tokens_map),
            supports_reasoning_budget=False,
            reports_reasoning_tokens=True,
            exposes_reasoning_content=True,
            supports_temperature=True,
            supports_seed=True,
            reasoning_included_in_output_tokens=False,
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        if request.reasoning_setting not in self.reasoning_tokens_map:
            raise ValueError(
                f"unsupported reasoning setting {request.reasoning_setting!r} "
                f"for mock model {self.model_id}"
            )
        start = time.monotonic()
        if request.reasoning_setting in self.fail_settings:
            time.sleep(0.01)
            return GenerationResult(
                provider="mock",
                model=self.model_id,
                model_version="mock-1.0",
                requested_reasoning_setting=request.reasoning_setting,
                effective_reasoning_setting=request.reasoning_setting,
                response_text="",
                error="mock failure",
                latency_ms=int((time.monotonic() - start) * 1000),
                temperature_requested=request.temperature_requested,
                temperature_effective=request.temperature_requested,
                seed_requested=request.seed_requested,
                seed_effective=request.seed_requested,
            )
        rt = self.reasoning_tokens_map[request.reasoning_setting]
        text = self.response_text
        time.sleep(0.01)
        return GenerationResult(
            provider="mock",
            model=self.model_id,
            model_version="mock-1.0",
            requested_reasoning_setting=request.reasoning_setting,
            effective_reasoning_setting=request.reasoning_setting,
            response_text=text,
            parsed_answer="42",
            input_tokens=100,
            reasoning_tokens=rt,
            output_tokens=len(text) // 4,
            total_tokens=100 + rt + len(text) // 4,
            latency_ms=int((time.monotonic() - start) * 1000),
            provider_request_id="mock-req-1",
            finish_reason="stop",
            temperature_requested=request.temperature_requested,
            temperature_effective=request.temperature_requested,
            seed_requested=request.seed_requested,
            seed_effective=request.seed_requested,
            cost_usd=0.0,
        )

