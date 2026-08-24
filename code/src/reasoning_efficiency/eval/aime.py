"""AIME deterministic evaluator: exact integer answer match."""

from __future__ import annotations

import re

from .math import extract_answer, extract_boxed


def extract_aime_answer(response: str) -> int | None:
    boxed = extract_boxed(response)
    if boxed:
        m = re.search(r"\d+", boxed)
        if m:
            return int(m.group(0))
    extracted = extract_answer(response)
    m = re.search(r"\d+", extracted)
    if m:
        return int(m.group(0))
    # fallback: last standalone integer in the response
    matches = re.findall(r"(?<!\d)(\d{1,20})(?!\d)", response)
    if matches:
        return int(matches[-1])
    return None


def evaluate_aime(response: str, reference: str) -> bool:
    pred = extract_aime_answer(response)
    if pred is None:
        return False
    try:
        ref = int(str(reference).strip())
    except ValueError:
        return False
    return pred == ref


def evaluate_text_answer(response: str, reference: str) -> bool:
    """Normalized exact-text answer match (MechanismProbe logic items)."""
    import re as _re

    boxed = extract_boxed(response)
    if boxed:
        pred = boxed
    else:
        m = _re.search(
            r"(?:the\s+)?(?:answer|答案)\s*(?:is\s+|:|=)?\s*(.+?)\s*$",
            response,
            flags=_re.IGNORECASE | _re.MULTILINE,
        )
        pred = m.group(1) if m else response.strip()
    norm = lambda s: _re.sub(r"\s+", " ", s.strip().lower()).rstrip(".")
    return norm(pred) == norm(reference)
