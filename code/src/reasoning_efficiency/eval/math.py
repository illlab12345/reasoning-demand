"""Deterministic math answer normalization and equivalence checking."""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

TRANSFORMATIONS = standard_transformations + (convert_xor, implicit_multiplication_application)

BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


def extract_boxed(text: str) -> str | None:
    m = BOXED_RE.search(text)
    return m.group(1).strip() if m else None


def normalize_latex(text: str) -> str:
    s = text.strip()
    s = re.sub(r"\$", "", s)
    s = re.sub(r"\\(?:left|right|,|;|!|:|quad|qquad|\s)", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def strip_equation(s: str) -> str:
    s = s.strip()
    m = re.match(r"^([A-Za-z]\w*)\s*=\s*(.+)$", s)
    return m.group(2).strip() if m else s


def _find_matching_brace(s: str, start: int) -> int:
    """Return index of the brace matching s[start] (which must be '{')."""
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def latex_to_plain(s: str) -> str:
    """Convert common LaTeX math constructs into a parse_expr-friendly string.

    Deterministic and dependency-light: handles \\frac, \\sqrt, ^{...}, _{...},
    common operators and spacing. Unsupported constructs degrade to string
    comparison in math_answers_equal.
    """
    s = s.strip()
    s = re.sub(r"\$", "", s)
    s = re.sub(r"\\(?:left|right|,|;|!|:|\s)", "", s)

    while "\\frac" in s:
        idx = s.index("\\frac")
        i = idx + 5
        while i < len(s) and s[i] == " ":
            i += 1
        if i >= len(s) or s[i] != "{":
            break
        end1 = _find_matching_brace(s, i)
        if end1 < 0:
            break
        j = end1 + 1
        while j < len(s) and s[j] == " ":
            j += 1
        if j >= len(s) or s[j] != "{":
            break
        end2 = _find_matching_brace(s, j)
        if end2 < 0:
            break
        num, den = s[i + 1 : end1], s[j + 1 : end2]
        s = s[:idx] + f"({num})/({den})" + s[end2 + 1 :]

    s = re.sub(r"\\sqrt\[([^{}]*)\]\{([^{}]*)\}", r"(\2)**(1/(\1))", s)
    s = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", s)
    s = re.sub(r"\^\{([^{}]*)\}", r"**(\1)", s)
    s = re.sub(r"\^([A-Za-z0-9.+\-]+)", r"**(\1)", s)
    s = re.sub(r"_\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"_([A-Za-z0-9]+)", r"\1", s)
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    s = re.sub(r"\{", "(", s).replace("}", ")")
    s = re.sub(r"\s+", "", s)
    return s


def _sympy_equal(a: str, b: str) -> bool:
    a, b = strip_equation(a), strip_equation(b)
    plain_a, plain_b = latex_to_plain(a), latex_to_plain(b)
    try:
        ea = parse_expr(plain_a, transformations=TRANSFORMATIONS)
        eb = parse_expr(plain_b, transformations=TRANSFORMATIONS)
        return bool(sympy.simplify(ea - eb) == 0)
    except Exception:
        return False


def math_answers_equal(pred: str, ref: str) -> bool:
    if normalize_latex(pred) == normalize_latex(ref):
        return True
    return _sympy_equal(pred, ref)


def extract_answer(response: str) -> str:
    """Extract the final answer from a model response (deterministic heuristics)."""
    s = response.strip()
    if not s:
        return ""

    boxed = extract_boxed(s)
    if boxed:
        return boxed

    for pattern in (
        r"[Aa]nswer\s*(?:is|:|=)?\s*(.+?)(?:\n|$)",
        r"[Ff]inal\s+answer\s*(?:is|:|=)?\s*(.+?)(?:\n|$)",
        r"答案\s*[:：]?\s*(.+?)(?:\n|$)",
    ):
        m = re.search(pattern, s)
        if m:
            return m.group(1).strip().rstrip(".")

    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        last = re.sub(r"^[-*•]\s+", "", last)
        last = re.sub(r"^\(\s*[A-Za-z0-9]+\s*\)\s*", "", last)
        return last.rstrip(".")
    return ""


def evaluate_math(response: str, reference: str) -> bool:
    pred = extract_answer(response)
    if not pred:
        return False
    return math_answers_equal(pred, reference)
