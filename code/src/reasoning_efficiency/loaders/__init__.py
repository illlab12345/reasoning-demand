"""Dataset loaders: raw file(s) -> unified-schema records."""

from __future__ import annotations

from .base import LoadedDataset, first_present, make_jsonable
from .aime import load_aime
from .easy2hard import load_easy2hard
from .gpqa import load_gpqa
from .livecodebench import load_livecodebench
from .math500 import load_math500
from .zebralogic import load_zebralogic

LOADERS = {
    "aime": load_aime,
    "math500": load_math500,
    "zebralogic": load_zebralogic,
    "easy2hard": load_easy2hard,
    "gpqa": load_gpqa,
    "livecodebench": load_livecodebench,
}


def get_loader(name: str):
    if name not in LOADERS:
        raise KeyError(f"unknown loader: {name!r}; available: {sorted(LOADERS)}")
    return LOADERS[name]


__all__ = [
    "LOADERS",
    "LoadedDataset",
    "first_present",
    "get_loader",
    "load_aime",
    "load_easy2hard",
    "load_gpqa",
    "load_livecodebench",
    "load_math500",
    "load_zebralogic",
    "make_jsonable",
]
