"""Deterministic evaluators for pilot benchmarks."""

from __future__ import annotations

from typing import Any

from .math import evaluate_math
from .multiple_choice import evaluate_mc
from .zebra_grid import evaluate_grid
from .zebra import evaluate_zebra
from .aime import evaluate_aime, evaluate_text_answer
from .code_exec import evaluate_livecodebench


def evaluate_answer(record: dict[str, Any], response: str) -> bool | None:
    """Evaluate a model response against a processed record.

    Returns True/False when a deterministic evaluator exists for the dataset,
    None when the dataset/subset is not yet supported (e.g. code execution).
    """
    dataset = record.get("dataset")
    if dataset == "MATH-500":
        return evaluate_math(response, record["answer"])
    if dataset == "AIME":
        return evaluate_aime(response, record["answer"])
    if dataset == "MechanismProbe":
        if record.get("metadata", {}).get("evaluator") == "text":
            return evaluate_text_answer(response, record["answer"])
        return evaluate_aime(response, record["answer"])
    if dataset == "GPQA-Diamond":
        return evaluate_mc(
            response,
            record["answer"],
            options=record.get("metadata", {}).get("options"),
            n_options=4,
        )
    if dataset == "LiveCodeBench":
        return evaluate_livecodebench(response, record)
    if dataset == "ZebraLogicBench":
        mode = record.get("metadata", {}).get("mode", "mc")
        if mode == "grid":
            return evaluate_grid(record, response).correct
        return evaluate_zebra(
            response,
            record["answer"],
            options=record.get("metadata", {}).get("options"),
            mode=mode,
        )
    if dataset == "Easy2Hard-Bench":
        subset = record.get("metadata", {}).get("subset")
        if subset in ("E2H-AMC", "E2H-GSM8K"):
            return evaluate_math(response, record["answer"])
        if subset in ("E2H-ARC", "E2H-Winogrande"):
            return evaluate_mc(response, record["answer"], options=record.get("metadata", {}).get("options"))
        return None  # E2H-Codeforces / Lichess need execution/chess evaluators (later)
    return None


__all__ = [
    "evaluate_answer",
    "evaluate_aime",
    "evaluate_text_answer",
    "evaluate_grid",
    "evaluate_livecodebench",
    "evaluate_math",
    "evaluate_mc",
    "evaluate_zebra",
]
