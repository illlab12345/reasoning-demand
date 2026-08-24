#!/usr/bin/env python
"""Generate the state-tracking mechanism experiment (0 API).

Task: track k variables through 6 steps of independent arithmetic updates and
report the SUM of all final values. Manipulated factor: state-tracking load
k in {2, 4, 8} (number of variables that must be tracked simultaneously);
sequential depth is fixed at 6. Matched design: the k=2 and k=4 variants use
the first k variables and the corresponding operations of the k=8 base.

Predicted endpoints (prespecified before data collection):
  E1 (r* shift): at k=8, item-level paired (high - low) one-sided 95%
     bootstrap lower bound > 0; at k=2 the bound <= 0 or difference small.
  E2 (token scaling): mean `low` reasoning tokens at k=8 > k=2.
  E3 (monotonicity): low accuracy k=2 >= k=4 >= k=8.

Outputs:
  datasets/probe/p1_statetrack_v1.jsonl (+ manifest)
  configs/p1_statetrack.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

from reasoning_efficiency.io import file_sha256, write_json, write_jsonl  # noqa: E402

SEED = 20260818
N_BASES = 20
K_LEVELS = [2, 4, 8]
STEPS = 6
NAMES = list("ABCDEFGH")
SETTINGS = ["low", "high"]


def _ops_for_variable(rng: random.Random) -> list[tuple[str, int]]:
    """Six operations for one variable, with at least two multiplications."""
    ops = []
    mult_positions = rng.sample(range(STEPS), 2)
    for i in range(STEPS):
        if i in mult_positions:
            ops.append(("x", rng.randint(2, 9)))
        else:
            ops.append(("+", rng.randint(11, 99)))
    return ops


def _item(base_seed: int, k: int, idx: int, with_max: bool) -> dict:
    rng = random.Random(base_seed * 1000 + k)
    # k=8 master: starts + per-variable ops
    starts8 = [rng.randint(1000, 9999) for _ in range(8)]
    ops8 = [_ops_for_variable(rng) for _ in range(8)]

    starts = starts8[:k]
    ops = ops8[:k]
    final = list(starts)
    for step in range(STEPS):
        for v in range(k):
            kind, val = ops[v][step]
            final[v] = final[v] * val if kind == "x" else final[v] + val
    answer = sum(final)

    var_lines = "; ".join(f"{NAMES[v]} = {starts[v]}" for v in range(k))
    step_lines = []
    for step in range(STEPS):
        parts = [f"{NAMES[v]}: {'x' if ops[v][step][0] == 'x' else '+'}{ops[v][step][1]}" for v in range(k)]
        step_lines.append(f"Step {step + 1}: " + "; ".join(parts))
    if k == 2:
        name_clause = "A and B"
    elif k > 2:
        name_clause = f"A through {NAMES[k-1]}"
    else:
        name_clause = "A"
    question = (
        f"You are tracking {k} variables ({name_clause}). "
        f"Their starting values are: {var_lines}.\n"
        "Apply the following operations in order. Each step updates ALL variables using that step's rule.\n"
        + "\n".join(step_lines)
        + f"\nAfter all {STEPS} steps, what is the SUM of the final values of all {k} variables? "
        'Provide the final answer in a line starting with "Answer:".'
    )
    settings = list(SETTINGS) + (["max"] if with_max else [])
    return {
        "id": f"st_base{base_seed:08d}_k{k}_{idx:03d}",
        "dataset": "MechanismProbe",
        "domain": "reasoning",
        "stratum": f"k{k}",
        "base_seed": base_seed,
        "n_variables": k,
        "depth": STEPS,
        "question": question,
        "answer": str(answer),
        "difficulty": None,
        "metadata": {
            "factor": "state_tracking",
            "k": k,
            "depth": STEPS,
            "start_values": starts,
            "steps": ops,
            "evaluator": "int",
        },
        "_probe_settings": settings,
        "_prompt": "aime_v1",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "probe")
    ap.add_argument("--n-bases", type=int, default=N_BASES)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    items = []
    max_items = 0
    for b in range(args.n_bases):
        base_seed = SEED + b
        for k in K_LEVELS:
            with_max = k == K_LEVELS[-1]
            max_items += int(with_max)
            items.append(_item(base_seed, k, b, with_max=with_max))

    path = args.out / "p1_statetrack_v1.jsonl"
    write_jsonl(path, items)
    write_json(
        path.with_suffix(".manifest.json"),
        {
            "file": path.name,
            "records": len(items),
            "seed": SEED,
            "factor": "state_tracking",
            "k_levels": K_LEVELS,
            "depth": STEPS,
            "bases": args.n_bases,
            "max_validation_variants": max_items,
            "file_sha256": file_sha256(path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    cfg = {
        "schema_version": "p1-statetrack-v1",
        "model": "deepseek-v4-flash",
        "repeats": 3,
        "temperature": 0.0,
        "seed": None,
        "item_files": [str(path)],
        "primary_endpoints": {
            "r_shift_at_k8": "one-sided 95% item-level bootstrap lower bound of (high-low) accuracy at k=8 > 0",
            "token_scaling": "mean low reasoning tokens at k=8 > k=2",
            "monotonicity": "low accuracy k=2 >= k=4 >= k=8",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "code" / "configs" / "p1_statetrack.yaml", cfg)
    print(f"items: {len(items)} -> {path}")
    print("max validation items:", max_items)
    print("sample question (k=2):")
    print(items[0]["question"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
