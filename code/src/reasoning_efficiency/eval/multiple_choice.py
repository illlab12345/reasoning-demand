"""Deterministic multiple-choice evaluator."""

from __future__ import annotations

import re
from typing import Any


def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def normalize_letter(s: str) -> str:
    return s.strip().upper().rstrip(".")


def _options_as_list(options: Any) -> list[str]:
    if isinstance(options, list):
        return [str(o) for o in options]
    if isinstance(options, dict):
        # either {"A": "text"} or {"label": [...], "text": [...]}
        if "label" in options and "text" in options:
            return [f"{l}. {t}" for l, t in zip(options["label"], options["text"])]
        out: list[str] = []
        for k in sorted(options):
            out.append(str(options[k]))
        return out
    return []


def _option_text_for_letter(options: Any, letter: str) -> str | None:
    if isinstance(options, list):
        idx = ord(letter) - ord("A")
        return str(options[idx]) if 0 <= idx < len(options) else None
    if isinstance(options, dict):
        if "label" in options and "text" in options:
            labels = [str(x) for x in options["label"]]
            texts = [str(x) for x in options["text"]]
            for l, t in zip(labels, texts):
                if l.upper() == letter:
                    return t
            return None
        return str(options.get(letter)) if letter in options else None
    return None


def extract_choice(response: str, n_options: int = 5, options: Any = None) -> str | None:
    """Extract the selected option letter from a response."""
    letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options])
    s = response.strip()
    if not s:
        return None

    # 1) response is exactly a letter
    if s.upper() in letters:
        return s.upper()

    # 1b) response is a 1-based option index ("1" -> "A", "2." -> "B")
    numeric = s.rstrip(".").strip()
    if numeric in {str(i) for i in range(1, n_options + 1)}:
        return chr(ord("A") + int(numeric) - 1)

    # 2) full text equals one of the option texts
    opt_list = _options_as_list(options) if options is not None else []
    for item in opt_list:
        m = re.match(r"^\(?([A-Za-z])\)?[.)\s:]*\s*(.+)$", item)
        if m and _norm_text(m.group(2)) == _norm_text(s):
            return m.group(1).upper()
        if _norm_text(item) == _norm_text(s):
            return None  # exact text but no letter available; caller can compare text

    # 3) "(A)" / "A." / "A:" / "A)" style markers
    for pattern in (
        r"\(([A-Za-z])\)",
        r"(?:^|\s)([A-Za-z])\s*[.):]\s*(?:\s|$)",
        r"(?:^|\s)option\s+([A-Za-z])\b",
        r"(?:^|\s)choice\s+([A-Za-z])\b",
        r"\b([A-Za-z])\s+is\s+(?:the\s+)?(?:correct|answer|right)\b",
        r"(?:answer|答案)\s*(?::|：|is|=)?\s*\(?([A-Za-z])\)?[.)]?\s*$",
    ):
        m = re.search(pattern, s, flags=re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter == "A" and pattern.startswith(r"(?:answer"):
                # guard: "Answer: The option is D" -> "T" is not the choice
                prefix = s[: m.start(1)]
                if any(w in prefix.lower() for w in ("option", "choice", "correct", "answer")) and ":" not in prefix and not prefix.strip().endswith(("is", "=")):
                    continue
            if letter in letters:
                return letter
    return None


def _reference_letter(reference: str, n_options: int) -> str | None:
    ref = normalize_letter(reference)
    letters = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"[:n_options])
    if ref in letters:
        return ref
    if ref in {str(i) for i in range(1, n_options + 1)}:
        return chr(ord("A") + int(ref) - 1)
    return None


def evaluate_mc(response: str, reference: str, options: Any = None, n_options: int = 5) -> bool:
    """Evaluate a multiple-choice response.

    `reference` may be an option letter ("B") or a 1-based option index ("2").
    When `options` is provided, exact text matching is also accepted.
    """
    ref_letter = _reference_letter(reference, n_options)
    if ref_letter is None:
        return False
    pred = extract_choice(response, n_options=n_options, options=options)
    if pred == ref_letter:
        return True
    ref_text = _option_text_for_letter(options, ref_letter) if options is not None else None
    if ref_text is not None and _norm_text(ref_text) == _norm_text(response):
        return True
    return False
