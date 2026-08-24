from __future__ import annotations

import json

from reasoning_efficiency.eval import evaluate_answer
from reasoning_efficiency.eval.aime import evaluate_aime, extract_aime_answer
from reasoning_efficiency.eval.code_exec import extract_code, run_tests
from reasoning_efficiency.loaders.gpqa import _parse_query


def test_aime_extract_and_eval():
    assert extract_aime_answer("Let me compute...\nAnswer: 204") == 204
    assert extract_aime_answer(r"Final: \boxed{204}") == 204
    assert extract_aime_answer("no number here") is None
    assert evaluate_aime("Answer: 204", "204") is True
    assert evaluate_aime("Answer: 205", "204") is False
    assert evaluate_aime("some reasoning 12 then answer 42", "42") is True


def test_gpqa_parse_query():
    q = "What is x?\n\nChoices:\nA: 1\nB: 2\nC: 3\nD: 4"
    question, options = _parse_query(q)
    assert question == "What is x?"
    assert options == {"A": "1", "B": "2", "C": "3", "D": "4"}


def test_gpqa_evaluate_answer():
    record = {
        "dataset": "GPQA-Diamond",
        "answer": "D",
        "metadata": {"options": {"A": "1", "B": "2", "C": "3", "D": "4"}},
    }
    assert evaluate_answer(record, "D") is True
    assert evaluate_answer(record, "The answer is B") is False


def test_extract_code_from_markdown():
    text = "Here is my solution:\n```python\nprint(1)\n```\n"
    assert extract_code(text) == "print(1)"
    assert extract_code("print(2)") == "print(2)"


def test_run_tests_pass_and_fail():
    code = "import sys\nprint(int(sys.stdin.read().strip()) * 2)"
    tests = [{"input": "4", "output": "8"}, {"input": "21", "output": "42"}]
    result = run_tests(code, tests)
    assert result.correct is True
    assert result.n_passed == 2

    bad = [{"input": "4", "output": "9"}]
    result2 = run_tests(code, bad)
    assert result2.correct is False
    assert result2.n_passed == 0


def test_livecodebench_evaluate_answer():
    record = {
        "dataset": "LiveCodeBench",
        "answer": json.dumps({"public": [{"input": "4", "output": "8"}], "private": []}),
        "metadata": {},
    }
    ok_response = "```python\nimport sys\nprint(int(sys.stdin.read().strip()) * 2)\n```"
    bad_response = "```python\nprint(1)\n```"
    assert evaluate_answer(record, ok_response) is True
    assert evaluate_answer(record, bad_response) is False

