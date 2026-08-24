"""LiveCodeBench deterministic code-execution evaluator.

Runs the extracted Python program against the problem's public and private
test cases in a subprocess with per-test and total timeouts. Output comparison
is exact after stripping surrounding whitespace (documented normalization).
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CodeEvalResult:
    correct: bool
    n_passed: int
    n_total: int
    error: str | None = None
    failed_examples: list[dict[str, Any]] = field(default_factory=list)


CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    blocks = CODE_BLOCK_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()
    return text.strip()


def _normalize_output(s: str) -> str:
    return s.strip()


def run_tests(
    code: str,
    tests: list[dict[str, Any]],
    timeout_per_test: float = 8.0,
    max_total_seconds: float = 120.0,
) -> CodeEvalResult:
    if not code:
        return CodeEvalResult(False, 0, len(tests), error="empty code")
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "solution.py"
        script.write_text(code, encoding="utf-8")
        start = time.monotonic()
        n_passed = 0
        failed: list[dict[str, Any]] = []
        for i, t in enumerate(tests):
            if not isinstance(t, dict):
                continue
            inp = t.get("input", "")
            expected = t.get("output", "")
            remaining = max(0.5, max_total_seconds - (time.monotonic() - start))
            try:
                proc = subprocess.run(
                    [sys.executable, str(script)],
                    input=str(inp),
                    capture_output=True,
                    text=True,
                    timeout=min(timeout_per_test, remaining),
                    cwd=td,
                )
                if proc.returncode != 0:
                    raise RuntimeError(f"exit code {proc.returncode}: {proc.stderr[:300]}")
                if _normalize_output(proc.stdout) == _normalize_output(str(expected)):
                    n_passed += 1
                else:
                    failed.append({"case": i, "expected": str(expected)[:200], "got": proc.stdout[:200]})
            except subprocess.TimeoutExpired:
                failed.append({"case": i, "error": "timeout"})
            except Exception as e:  # noqa: BLE001
                failed.append({"case": i, "error": str(e)[:200]})
            if time.monotonic() - start >= max_total_seconds:
                failed.append({"case": "truncated", "error": "total time budget exceeded"})
                break
        total = len(tests)
        return CodeEvalResult(
            correct=n_passed == total and not failed,
            n_passed=n_passed,
            n_total=total,
            error=None if n_passed == total else f"{total - n_passed} test(s) failed",
            failed_examples=failed[:5],
        )


def evaluate_livecodebench(response: str, record: dict[str, Any]) -> bool:
    try:
        tests = json.loads(record["answer"])
    except (TypeError, json.JSONDecodeError):
        return False
    all_tests = list(tests.get("public", [])) + list(tests.get("private", []))
    if not all_tests:
        return False
    code = extract_code(response)
    return run_tests(code, all_tests).correct

