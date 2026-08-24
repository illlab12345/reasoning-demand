"""Answer parsing per benchmark (thin layer over evaluator extractors)."""

from __future__ import annotations

from typing import Any

from .eval.math import extract_answer
from .eval.multiple_choice import extract_choice
from .eval.aime import extract_aime_answer
from .eval.code_exec import extract_code


def parse_answer_for_record(record: dict[str, Any], response_text: str) -> str | None:
    """Extract a short answer string for logging; evaluation still uses evaluators."""
    dataset = record.get("dataset")
    if dataset == "MATH-500":
        return extract_answer(response_text)
    if dataset == "AIME":
        a = extract_aime_answer(response_text)
        return str(a) if a is not None else None
    if dataset == "GPQA-Diamond":
        return extract_choice(
            response_text,
            n_options=4,
            options=record.get("metadata", {}).get("options"),
        )
    if dataset == "LiveCodeBench":
        return extract_code(response_text)[:500]
    if dataset == "Easy2Hard-Bench":
        subset = record.get("metadata", {}).get("subset")
        if subset in ("E2H-AMC", "E2H-GSM8K"):
            return extract_answer(response_text)
        if subset in ("E2H-ARC", "E2H-Winogrande"):
            return extract_choice(
                response_text,
                n_options=len(record.get("metadata", {}).get("options") or []) or 5,
                options=record.get("metadata", {}).get("options"),
            )
    if dataset == "ZebraLogicBench" and record.get("metadata", {}).get("mode") == "mc":
        options = record.get("metadata", {}).get("options") or []
        return extract_choice(response_text, n_options=len(options) or 5, options=options)
    return response_text.strip() or None
