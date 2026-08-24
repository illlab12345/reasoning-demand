from __future__ import annotations

from reasoning_efficiency.eval import evaluate_answer, evaluate_math, evaluate_mc, evaluate_zebra
from reasoning_efficiency.eval.math import extract_answer, extract_boxed, math_answers_equal
from reasoning_efficiency.eval.multiple_choice import extract_choice


def test_math_extract_boxed():
    assert extract_boxed(r"The result is \boxed{42}.") == "42"
    assert extract_boxed(r"Final: \boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_math_extract_answer_heuristics():
    assert extract_answer("Let me think...\nAnswer: 42") == "42"
    assert extract_answer("The answer is 1/2.") == "1/2"
    assert extract_answer("Some reasoning\n\n- 7") == "7"


def test_math_equivalence():
    assert math_answers_equal(r"\frac{1}{2}", "0.5")
    assert math_answers_equal("1/2", "0.5")
    assert math_answers_equal("x = 3", "3")
    assert not math_answers_equal("2", "3")


def test_evaluate_math():
    assert evaluate_math(r"The answer is \boxed{5}", "5")
    assert not evaluate_math("The answer is 6", "5")


def test_mc_extract_choice():
    assert extract_choice("A") == "A"
    assert extract_choice("(B)") == "B"
    assert extract_choice("The answer is C.") == "C"
    assert extract_choice("Option D") == "D"
    assert extract_choice("I believe E is correct") == "E"


def test_mc_evaluate():
    assert evaluate_mc("B", "B")
    assert not evaluate_mc("C", "B")
    assert evaluate_mc("The correct answer is (A)", "A")
    assert evaluate_mc("Answer: D", "D")
    assert evaluate_mc("The answer is B", "B")
    assert evaluate_mc("答案：C", "C")


def test_zebra_mc():
    options = ["Eric", "Bob", "Alice", "Peter", "Carol", "Arnold"]
    assert evaluate_zebra("Bob", "Bob", options=options, mode="mc")
    assert evaluate_zebra("The answer is Bob", "Bob", options=options, mode="mc")
    assert evaluate_zebra("B", "Bob", options=options, mode="mc")
    assert not evaluate_zebra("Alice", "Bob", options=options, mode="mc")
    assert not evaluate_zebra("A", "Bob", options=options, mode="mc")


def test_evaluate_answer_dispatch_math():
    record = {
        "dataset": "MATH-500",
        "answer": r"\frac{1}{2}",
        "metadata": {},
    }
    assert evaluate_answer(record, "0.5") is True
    assert evaluate_answer(record, "3") is False


def test_evaluate_answer_dispatch_zebra():
    record = {
        "dataset": "ZebraLogicBench",
        "answer": "Bob",
        "metadata": {"mode": "mc", "options": ["Eric", "Bob", "Alice", "Peter", "Carol", "Arnold"]},
    }
    assert evaluate_answer(record, "B") is True
    assert evaluate_answer(record, "The answer is Bob") is True


def test_mc_option_index_reference():
    options = ["Natalie", "Lindsey"]
    assert evaluate_mc("1", "1", options=options, n_options=2)
    assert evaluate_mc("A", "1", options=options, n_options=2)
    assert evaluate_mc("Natalie", "1", options=options, n_options=2)
    assert not evaluate_mc("Lindsey", "1", options=options, n_options=2)


def test_mc_arc_dict_options():
    options = {"A": "red", "B": "blue", "C": "green", "D": "yellow"}
    assert evaluate_mc("B", "B", options=options, n_options=4)
    assert evaluate_mc("blue", "B", options=options, n_options=4)
    assert not evaluate_mc("green", "B", options=options, n_options=4)


def test_evaluate_answer_dispatch_easy2hard_unsupported():
    record = {
        "dataset": "Easy2Hard-Bench",
        "answer": "x",
        "metadata": {"subset": "E2H-Codeforces"},
    }
    assert evaluate_answer(record, "anything") is None
