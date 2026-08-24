"""Cost estimation using configs/pricing.yaml."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class CostEstimate:
    status: str  # ok | missing_prices
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float | None = None
    note: str = ""


def _get(result: Any, key: str):
    if isinstance(result, dict):
        return result.get(key)
    return getattr(result, key, None)


def request_cost_usd(result: Any, price: dict[str, Any], reasoning_included_in_output: bool | None = None) -> float | None:
    """Cost of a single GenerationResult. None if prices are missing."""
    input_price = price.get("input_cache_miss")
    output_price = price.get("output")
    if input_price is None or output_price is None:
        return None
    input_tokens = _get(result, "input_tokens") or 0
    output_tokens = _get(result, "output_tokens") or 0
    reasoning_tokens = _get(result, "reasoning_tokens") or 0
    if reasoning_included_in_output is False and reasoning_tokens:
        billable_output = output_tokens + reasoning_tokens
    else:
        billable_output = output_tokens
    return input_tokens / 1e6 * input_price + billable_output / 1e6 * output_price


def estimate(results: list[Any], prices: dict[str, Any], reasoning_included: bool | None = None) -> CostEstimate:
    if not results:
        return CostEstimate(status="ok", note="no requests")
    error_count = sum(1 for r in results if _get(r, "error"))
    if error_count:
        return CostEstimate(
            status="errors",
            requests=len(results),
            input_tokens=sum(_get(r, "input_tokens") or 0 for r in results),
            output_tokens=sum(_get(r, "output_tokens") or 0 for r in results),
            reasoning_tokens=sum(_get(r, "reasoning_tokens") or 0 for r in results),
            cost_usd=None,
            note=f"{error_count}/{len(results)} sampled requests failed; fix adapter before estimating",
        )
    price = next(iter(prices.values())) if isinstance(prices, dict) and prices else {}
    if isinstance(prices, dict) and len(prices) == 1:
        price = next(iter(prices.values()))
    elif isinstance(prices, dict):
        # use first model's price for estimation (documented approximation)
        price = next(iter(prices.values()))
    input_tokens = sum(_get(r, "input_tokens") or 0 for r in results)
    output_tokens = sum(_get(r, "output_tokens") or 0 for r in results)
    reasoning_tokens = sum(_get(r, "reasoning_tokens") or 0 for r in results)
    costs = [request_cost_usd(r, price, reasoning_included) for r in results]
    if any(c is None for c in costs):
        return CostEstimate(
            status="missing_prices",
            requests=len(results),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            cost_usd=None,
            note="pricing.yaml has missing prices; fill before formal runs",
        )
    return CostEstimate(
        status="ok",
        requests=len(results),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cost_usd=round(sum(c for c in costs if c is not None), 6),
    )


def extrapolate(sample: CostEstimate, sample_requests: int, total_requests: int) -> CostEstimate:
    if sample_requests <= 0 or total_requests <= 0:
        return CostEstimate(status=sample.status, note="nothing to extrapolate")
    ratio = total_requests / sample_requests
    cost = sample.cost_usd * ratio if sample.cost_usd is not None else None
    return CostEstimate(
        status=sample.status,
        requests=total_requests,
        input_tokens=int(sample.input_tokens * ratio),
        output_tokens=int(sample.output_tokens * ratio),
        reasoning_tokens=int(sample.reasoning_tokens * ratio),
        cost_usd=round(cost, 2) if cost is not None else None,
        note=f"extrapolated from {sample_requests} sampled requests",
    )
