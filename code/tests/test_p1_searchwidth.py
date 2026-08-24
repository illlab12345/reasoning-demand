from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from reasoning_efficiency.eval import evaluate_answer

ROOT = Path(__file__).resolve().parents[2]


def _run_setup():
    out = subprocess.run(
        [sys.executable, str(ROOT / "code" / "scripts" / "p1_searchwidth_setup.py")],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert out.returncode == 0, out.stderr


def test_setup_counts_and_matching():
    _run_setup()
    smoke = [json.loads(l) for l in open(ROOT / "datasets" / "probe" / "p1_searchwidth_smoke_v3.jsonl", encoding="utf-8")]
    full = [json.loads(l) for l in open(ROOT / "datasets" / "probe" / "p1_searchwidth_v3.jsonl", encoding="utf-8")]
    assert len(smoke) == 30
    assert len(full) == 180
    # matched: 3 variants per base sharing base_seed and depth
    for items in (smoke, full):
        by_base = {}
        for it in items:
            by_base.setdefault(it["base_seed"], set()).add(it["branching"])
        assert all(v == {2, 4, 8} for v in by_base.values())
        assert all(it["depth"] == 6 for it in items)
        assert all(int(it["answer"]) > 0 for it in items)
    # max validation on exactly 60 full variants
    assert sum(1 for it in full if "max" in it["_probe_settings"]) == 60
    assert sum(1 for it in smoke if "max" in it["_probe_settings"]) == 0


def test_searchwidth_evaluator():
    item = {"dataset": "MechanismProbe", "answer": "32768", "metadata": {"evaluator": "int"}}
    assert evaluate_answer(item, "Answer: 32768") is True
    assert evaluate_answer(item, "Answer: 32767") is False
