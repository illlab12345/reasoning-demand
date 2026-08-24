from __future__ import annotations

import json

import pytest

from reasoning_efficiency.eval import evaluate_answer, evaluate_grid
from reasoning_efficiency.eval.zebra_grid import parse_ground_truth, parse_model_output


def _record(solution: dict) -> dict:
    return {
        "dataset": "ZebraLogicBench",
        "answer": json.dumps(solution, sort_keys=True),
        "metadata": {"mode": "grid", "solution": solution},
    }


GT = {
    "header": ["House", "Name", "Color", "Pet"],
    "rows": [
        ["1", "Alice", "red", "cat"],
        ["2", "Bob", "blue", "dog"],
        ["3", "Carol", "green", "bird"],
    ],
}


MARKDOWN = """
| House | Name | Color | Pet |
|-------|------|-------|-----|
| 1 | Alice | red | cat |
| 2 | Bob | blue | dog |
| 3 | Carol | green | bird |
"""


def test_parse_ground_truth():
    gt = parse_ground_truth(_record(GT))
    assert gt[("1", "name")] == "alice"
    assert len(gt) == 9


def test_markdown_exact_match():
    result = evaluate_grid(_record(GT), MARKDOWN)
    assert result.correct is True
    assert result.cell_accuracy == 1.0
    assert result.parse_error is None


def test_markdown_order_insensitive():
    reversed_md = """
| House | Name | Color | Pet |
|-------|------|-------|-----|
| 3 | Carol | green | bird |
| 2 | Bob | blue | dog |
| 1 | Alice | red | cat |
"""
    result = evaluate_grid(_record(GT), reversed_md)
    assert result.correct is True


def test_json_object_match():
    payload = json.dumps({"1": {"Name": "Alice", "Color": "red", "Pet": "cat"},
                          "2": {"Name": "Bob", "Color": "blue", "Pet": "dog"},
                          "3": {"Name": "Carol", "Color": "green", "Pet": "bird"}})
    result = evaluate_grid(_record(GT), payload)
    assert result.correct is True


def test_solution_json_match():
    result = evaluate_grid(_record(GT), json.dumps(GT))
    assert result.correct is True


def test_text_lines_match():
    text = "1: Name=Alice, Color=red, Pet=cat\n2: Name=Bob, Color=blue, Pet=dog\n3: Name=Carol, Color=green, Pet=bird"
    result = evaluate_grid(_record(GT), text)
    assert result.correct is True


def test_one_wrong_cell():
    wrong = MARKDOWN.replace("| 2 | Bob | blue | dog |", "| 2 | Bob | blue | cat |")
    result = evaluate_grid(_record(GT), wrong)
    assert result.correct is False
    assert result.n_matched == 8
    assert result.n_cells == 9
    assert abs(result.cell_accuracy - 8 / 9) < 1e-9


def test_parse_failure():
    result = evaluate_grid(_record(GT), "I think the answer is Bob lives in house 2. No table here.")
    assert result.correct is False
    assert result.parse_error is not None
    assert result.n_matched == 0


def test_normalization_quotes_and_case():
    weird = MARKDOWN.replace("Alice", "“Alice”").replace("red", "RED")
    result = evaluate_grid(_record(GT), weird)
    assert result.correct is True


def test_evaluate_answer_grid_dispatch():
    assert evaluate_answer(_record(GT), MARKDOWN) is True
    assert evaluate_answer(_record(GT), "garbage output") is False


def test_parse_model_output_rejects_prose():
    with pytest.raises(ValueError):
        parse_model_output("just some prose without a table")
