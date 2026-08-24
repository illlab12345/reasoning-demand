"""OpenAI-compatible adapter (Responses API / Chat Completions).

DeepSeek is configured with wire_api=responses. Reasoning-token reporting is
extracted defensively from several possible usage shapes; anything not present
is recorded as None (never guessed).
"""

from __future__ import annotations

import os
import time
from typing import Any

from openai import OpenAI

from .base import GenerationRequest, GenerationResult, ModelCapabilities


def _first(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None


def _extract_usage(usage: Any, wire_api: str) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage = usage.dict()
    if not isinstance(usage, dict):
        return {}

    input_tokens = _first(usage.get("input_tokens"), usage.get("prompt_tokens"))
    output_tokens = _first(usage.get("output_tokens"), usage.get("completion_tokens"))
    total_tokens = _first(usage.get("total_tokens"))
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens

    details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    if isinstance(details, dict):
        reasoning_tokens = _first(
            details.get("reasoning_tokens"),
            usage.get("reasoning_tokens"),
        )
        reasoning_included = details.get("reasoning_tokens") is not None
    else:
        reasoning_tokens = usage.get("reasoning_tokens")
        reasoning_included = False
    return {
        "input_tokens": input_tokens,
        "reasoning_tokens": reasoning_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reasoning_included_in_output_tokens": reasoning_included,
        "raw": usage,
    }


def _extract_response_text(response: Any, wire_api: str) -> str:
    if wire_api == "responses":
        if hasattr(response, "output_text"):
            return response.output_text or ""
        parts = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "message":
                parts.extend(getattr(c, "text", "") or "" for c in (getattr(item, "content", []) or []))
        return "".join(parts)
    return getattr(response, "choices", [{}])[0].get("message", {}).get("content", "") if isinstance(
        getattr(response, "choices", None), list
    ) else ""


def build_responses_params(request: GenerationRequest, capabilities: ModelCapabilities) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": request.model,
        "input": request.prompt,
    }
    if capabilities.supports_reasoning_effort:
        params["reasoning"] = {"effort": request.reasoning_setting}
    if capabilities.supports_temperature and request.temperature_requested is not None:
        params["temperature"] = request.temperature_requested
    if capabilities.supports_seed and request.seed_requested is not None:
        params["seed"] = request.seed_requested
    if request.max_output_tokens is not None:
        params["max_output_tokens"] = request.max_output_tokens
    return params


class OpenAICompatibleAdapter:
    def __init__(
        self,
        model_id: str,
        provider_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
        capabilities: ModelCapabilities | None = None,
        api_key: str | None = None,
        wire_api: str | None = None,
    ):
        self.model_id = model_id
        self.provider_cfg = provider_cfg
        self.model_cfg = model_cfg
        self.wire_api = wire_api or provider_cfg.get("wire_api", "responses")
        base_url = provider_cfg.get("base_url")
        key = api_key or os.environ.get(provider_cfg.get("api_key_env", ""))
        if not key:
            raise RuntimeError(f"missing API key for provider {provider_cfg.get('name', 'unknown')}")
        self._client = OpenAI(base_url=base_url, api_key=key)
        self.capabilities = capabilities or ModelCapabilities(
            supports_reasoning_effort=bool(model_cfg.get("capabilities", {}).get("supports_reasoning_effort", False)),
            supported_reasoning_efforts=list(model_cfg.get("reasoning_settings", [])),
            supports_reasoning_budget=False,
        )

    def generate(self, request: GenerationRequest) -> GenerationResult:
        start = time.monotonic()
        try:
            if request.reasoning_setting not in self.capabilities.supported_reasoning_efforts:
                raise ValueError(
                    f"unsupported reasoning setting {request.reasoning_setting!r} for {self.model_id}"
                )
            if self.wire_api == "responses":
                params = build_responses_params(request, self.capabilities)
                raw = None
                last_error: Exception | None = None
                # capability negotiation: drop unsupported params (seed/temperature)
                # and retry once per param; effective_* reflect what actually shipped.
                for _ in range(4):
                    try:
                        raw = self._client.responses.create(**params)
                        break
                    except Exception as e:  # noqa: BLE001 - negotiation needs message inspection
                        last_error = e
                        message = str(e)
                        if "seed" in message and "seed" in params:
                            del params["seed"]
                            continue
                        if "temperature" in message and "temperature" in params:
                            del params["temperature"]
                            continue
                        raise
                if raw is None:
                    raise last_error or RuntimeError("responses.create failed")
            else:
                raise NotImplementedError("chat completions wire_api not implemented yet")

            usage = _extract_usage(getattr(raw, "usage", None), self.wire_api)
            text = _extract_response_text(raw, self.wire_api)
            latency_ms = int((time.monotonic() - start) * 1000)
            effective_temp = params.get("temperature", None)
            effective_seed = params.get("seed", None)
            output_items = getattr(raw, "output", None) or []
            output_types = [getattr(item, "type", None) for item in output_items]
            raw_metadata = {
                "output_types": output_types,
                "has_reasoning_item": any(t == "reasoning" for t in output_types),
                "status": getattr(raw, "status", None),
            }
            return GenerationResult(
                provider=self.provider_cfg.get("name", "unknown"),
                model=self.model_id,
                model_version=getattr(raw, "model", None) or self.model_id,
                requested_reasoning_setting=request.reasoning_setting,
                effective_reasoning_setting=request.reasoning_setting,
                response_text=text,
                input_tokens=usage.get("input_tokens"),
                reasoning_tokens=usage.get("reasoning_tokens"),
                output_tokens=usage.get("output_tokens"),
                total_tokens=usage.get("total_tokens"),
                latency_ms=latency_ms,
                provider_request_id=getattr(raw, "id", None),
                finish_reason=getattr(raw, "status", None),
                temperature_requested=request.temperature_requested,
                temperature_effective=effective_temp,
                seed_requested=request.seed_requested,
                seed_effective=effective_seed,
                raw_usage=usage.get("raw", {}),
                raw_metadata=raw_metadata,
            )
        except Exception as e:  # noqa: BLE001 - surface as GenerationResult.error
            return GenerationResult(
                provider=self.provider_cfg.get("name", "unknown"),
                model=self.model_id,
                model_version=self.model_id,
                requested_reasoning_setting=request.reasoning_setting,
                effective_reasoning_setting=request.reasoning_setting,
                response_text="",
                error=f"{type(e).__name__}: {e}",
                latency_ms=int((time.monotonic() - start) * 1000),
                temperature_requested=request.temperature_requested,
                seed_requested=request.seed_requested,
            )
