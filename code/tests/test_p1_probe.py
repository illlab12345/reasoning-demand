from __future__ import annotations

from pathlib import Path

from reasoning_efficiency.eval import evaluate_answer
from reasoning_efficiency.eval.aime import evaluate_text_answer

ROOT = Path(__file__).resolve().parents[2]


def test_mechanism_generator_deterministic_and_consistent():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, str(ROOT / "code" / "scripts" / "p1_probe_setup.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr

    import json

    items = [json.loads(l) for l in open(ROOT / "datasets" / "probe" / "mechanism_probe_v1.jsonl", encoding="utf-8")]
    assert len(items) == 30
    depths = {}
    for it in items:
        depths[it["metadata"]["depth"]] = depths.get(it["metadata"]["depth"], 0) + 1
    assert depths == {8: 10, 16: 10, 24: 10}

    # ground truth must match the chain computation (regenerate independently)
    v = items[0]["metadata"]["start"]
    for op, k in items[0]["metadata"]["steps"]:
        v = v + k if op == "+" else (v * k if op == "*" else max(1, v - k))
    assert int(items[0]["answer"]) == v


def test_mechanism_probe_evaluator():
    item = {
        "dataset": "MechanismProbe",
        "answer": "42",
        "metadata": {},
    }
    assert evaluate_answer(item, "Answer: 42") is True
    assert evaluate_answer(item, "Answer: 43") is False


def test_text_evaluator():
    assert evaluate_text_answer("Answer: cat", "cat") is True
    assert evaluate_text_answer("The answer is Dog.", "dog") is True
    assert evaluate_text_answer("Answer: fish", "cat") is False


def test_full_p1_setup_counts_and_consistency():
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, str(ROOT / "code" / "scripts" / "p1_full_setup.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr

    import json

    mech = [json.loads(l) for l in open(ROOT / "datasets" / "probe" / "p1_full_mechanism_v1.jsonl", encoding="utf-8")]
    prosp = [json.loads(l) for l in open(ROOT / "datasets" / "probe" / "p1_full_prospective_v1.jsonl", encoding="utf-8")]
    assert len(mech) == 180
    assert len(prosp) == 300
    assert len({r["id"] for r in mech}) == 180
    assert len({r["id"] for r in prosp}) == 300
    # 30 mechanism items should have max in probe settings
    assert sum(1 for r in mech if "max" in r["_probe_settings"]) == 30
    # exactly 60 prospective items get an additional max condition
    assert sum(1 for r in prosp if "max" in r["_probe_settings"] and r["_router_setting"] != "max") == 60
    # logic items use text evaluator, chain items use int
    logic = [r for r in mech if r["metadata"]["factor"] == "constraints"]
    chain = [r for r in mech if r["metadata"]["factor"] == "depth"]
    distractor = [r for r in mech if r["metadata"]["factor"] == "distractor"]
    assert len(logic) == 60 and len(chain) == 60 and len(distractor) == 60
    assert all(r["metadata"]["evaluator"] == "text" for r in logic)
    assert all(r["metadata"]["evaluator"] == "int" for r in chain)
