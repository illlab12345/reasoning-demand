"""Model adapters: unified generation interface."""

from __future__ import annotations

from .base import BaseAdapter, GenerationRequest, GenerationResult, ModelCapabilities
from .mock import MockAdapter
from .openai_compatible import OpenAICompatibleAdapter


def get_adapter(name: str, *args, **kwargs) -> BaseAdapter:
    if name == "mock":
        return MockAdapter(*args, **kwargs)
    if name == "deepseek":
        return OpenAICompatibleAdapter(*args, **kwargs)
    raise KeyError(f"unknown adapter: {name!r}")


__all__ = [
    "BaseAdapter",
    "GenerationRequest",
    "GenerationResult",
    "MockAdapter",
    "ModelCapabilities",
    "OpenAICompatibleAdapter",
    "get_adapter",
]

