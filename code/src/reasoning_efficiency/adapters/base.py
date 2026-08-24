"""Adapter interfaces: ModelCapabilities / GenerationRequest / GenerationResult."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelCapabilities:
    supports_reasoning_effort: bool
    supported_reasoning_efforts: list[str]
    supports_reasoning_budget: bool
    min_reasoning_budget: int | None = None
    max_reasoning_budget: int | None = None
    reports_reasoning_tokens: bool | None = None
    exposes_reasoning_content: bool | None = None
    supports_temperature: bool | None = None
    supports_seed: bool | None = None
    reasoning_included_in_output_tokens: bool | None = None

    @property
    def eligible_for_primary_art(self) -> bool:
        return bool(self.reports_reasoning_tokens)


@dataclass(frozen=True)
class GenerationRequest:
    provider: str
    model: str
    prompt: str
    reasoning_control_type: str
    reasoning_setting: str
    reasoning_budget_requested: int | None = None
    temperature_requested: float | None = None
    seed_requested: int | None = None
    max_output_tokens: int | None = None


@dataclass
class GenerationResult:
    provider: str
    model: str
    model_version: str | None = None
    requested_reasoning_setting: str | None = None
    effective_reasoning_setting: str | None = None
    response_text: str = ""
    parsed_answer: str | None = None
    input_tokens: int | None = None
    reasoning_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int | None = None
    provider_request_id: str | None = None
    finish_reason: str | None = None
    temperature_requested: float | None = None
    temperature_effective: float | None = None
    seed_requested: int | None = None
    seed_effective: int | None = None
    cost_usd: float = 0.0
    raw_response_path: str | None = None
    error: str | None = None
    raw_usage: dict[str, Any] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "model_version": self.model_version,
            "requested_reasoning_setting": self.requested_reasoning_setting,
            "effective_reasoning_setting": self.effective_reasoning_setting,
            "response_text": self.response_text,
            "parsed_answer": self.parsed_answer,
            "input_tokens": self.input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "provider_request_id": self.provider_request_id,
            "finish_reason": self.finish_reason,
            "temperature_requested": self.temperature_requested,
            "temperature_effective": self.temperature_effective,
            "seed_requested": self.seed_requested,
            "seed_effective": self.seed_effective,
            "cost_usd": self.cost_usd,
            "raw_response_path": self.raw_response_path,
            "error": self.error,
            "raw_usage": self.raw_usage,
            "raw_metadata": self.raw_metadata,
        }


class BaseAdapter(Protocol):
    model_id: str
    capabilities: ModelCapabilities

    def generate(self, request: GenerationRequest) -> GenerationResult:
        ...
