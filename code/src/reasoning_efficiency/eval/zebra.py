"""ZebraLogicBench deterministic evaluator."""

from __future__ import annotations

import re
from typing import Any

from .multiple_choice import _norm_text, extract_choice


def _strip_prefix(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^(?:the\s+)?(?:answer|correct answer)\s*(?:is|:|=)\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(?:我选|答案是|答案[:：]?)\s*", "", s)
    return s.strip().rstrip(".")


def evaluate_zebra(response: str, reference: str, options: Any = None, mode: str = "mc") -> bool:
    """Evaluate a ZebraLogic mc_mode response.

    The ground truth is the option *text* (e.g. "Bob"). Accepted model outputs:
    the exact text, the text with an "answer is" prefix, or the option letter
    that maps to the correct text (A/B/C/... by options order).
    grid_mode evaluation (structured grid comparison) is planned; it is not
    silently approximated with an LLM judge.
    """
    if mode != "mc":
        raise NotImplementedError(
            "grid_mode deterministic evaluator is not implemented yet; "
            "a structured grid comparison will be added before grid-mode inference."
        )

    ref_norm = _norm_text(reference)
    pred_norm = _norm_text(_strip_prefix(response))
    if pred_norm == ref_norm:
        return True

    if options:
        opt_list = [str(o) for o in options]
        letter = extract_choice(response, n_options=len(opt_list), options=opt_list)
        if letter:
            idx = ord(letter) - ord("A")
            if 0 <= idx < len(opt_list) and _norm_text(opt_list[idx]) == ref_norm:
                return True
        # response text equals a non-correct option -> False
        if any(_norm_text(o) == pred_norm for o in opt_list):
            return False
    return False
