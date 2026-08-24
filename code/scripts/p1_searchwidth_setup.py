#!/usr/bin/env python
"""Search-width mechanism experiment (zero API calls).

Task: count the number of directed paths of EXACTLY k steps from s to t in a
small random directed graph. Manipulated factor: branching factor B in {2,4,8}
(each node has exactly B outgoing edges); depth k=5 is fixed; matched pairs
share (s, t, k, base seed) across B. A guaranteed chain s->...->t of length k
is embedded, so every graph is solvable and the answer is a positive integer.

Pre-registered endpoints:
  E1 (R* shift): at B=8, item-level paired (high - low) accuracy difference has
     one-sided 95% bootstrap lower bound > 0; at B=2 the same bound is <= 0 or
     the difference is small.
  E2 (token scaling): mean `low` reasoning tokens at B=8 > B=2.
  E3 (monotonicity): low accuracy at B=2 >= B=4 >= B=8.

Outputs:
  datasets/probe/p1_searchwidth_smoke_v1.jsonl (10 bases x 3 B)
  datasets/probe/p1_searchwidth_v1.jsonl (60 bases x 3 B)
  configs/p1_searchwidth_smoke.yaml / p1_searchwidth.yaml
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

SEED = 20260817
N_NODES = 20
K = 6
B_LEVELS = [2, 4, 8]
SETTINGS = ["low", "high"]


def _graph(base_seed: int, B: int, depth: int = K) -> tuple[dict[int, list[int]], int, int, int]:
    """Return (edges, path_count, s, t) for a matched graph with branching B."""
    rng = random.Random(base_seed * 1000 + B)
    nodes = list(range(N_NODES))
    s = rng.randrange(N_NODES)
    t = rng.randrange(N_NODES - 1)
    if t >= s:
        t += 1
    interior = rng.sample([x for x in nodes if x not in (s, t)], depth - 1)
    chain = [s] + interior + [t]
    edges: dict[int, list[int]] = {i: [] for i in nodes}
    for a, b in zip(chain, chain[1:]):
        edges[a].append(b)
    for i in nodes:
        while len(edges[i]) < B:
            j = rng.randrange(N_NODES)
            if j != i and j not in edges[i]:
                edges[i].append(j)
    # DP count of paths of exactly K steps from s to t
    counts = {s: 1}
    for _ in range(depth):
        nxt = {i: 0 for i in nodes}
        for u, c in counts.items():
            for v in edges[u]:
                nxt[v] += c
        counts = nxt
    return edges, counts[t], s, t


def _item(base_seed: int, B: int, idx: int, with_max: bool, depth: int = K) -> dict:
    edges, count, s, t = _graph(base_seed, B, depth)
    lines = [f"Node {i}: {','.join(map(str, sorted(edges[i])))}" for i in range(N_NODES)]
    graph_text = "\n".join(lines)
    question = (
        f"Consider a directed graph with nodes 0..{N_NODES - 1}. From a node you can move along any of its "
        f"directed edges.\n{graph_text}\n"
        f"How many distinct paths of exactly {depth} steps start at node {s} and end at node {t}? "
        f"Provide the final answer in a line starting with \"Answer:\"."
    )
    settings = list(SETTINGS) + (["max"] if with_max else [])
    return {
        "id": f"sw_base{base_seed:04d}_B{B}_{idx:03d}",
        "dataset": "MechanismProbe",
        "domain": "reasoning",
        "stratum": f"B{B}",
        "base_seed": base_seed,
        "branching": B,
        "depth": depth,
        "question": question,
        "answer": str(count),
        "difficulty": None,
        "metadata": {"factor": "search_width", "branching": B, "depth": depth, "path_count": count, "evaluator": "int"},
        "_probe_settings": settings,
        "_prompt": "aime_v1",
    }


def _generate(n_bases: int, max_on_every_base: bool, depth: int = K) -> list[dict]:
    items = []
    for b in range(n_bases):
        base_seed = SEED + b
        for B in B_LEVELS:
            items.append(_item(base_seed, B, b, with_max=max_on_every_base, depth=depth))
    return items


def _config(item_file: Path, n_bases: int, version: str) -> dict:
    return {
        "schema_version": version,
        "model": "deepseek-v4-flash",
        "repeats": 3,
        "temperature": 0.0,
        "seed": None,
        "item_files": [str(item_file)],
        "primary_endpoints": {
            "r_shift_at_B8": "one-sided 95% item-level bootstrap lower bound of (high-low) accuracy at B=8 > 0",
            "token_scaling": "mean low reasoning tokens at B=8 > B=2",
            "monotonicity": "low accuracy B=2 >= B=4 >= B=8",
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "probe")
    ap.add_argument("--depth", type=int, default=K)
    ap.add_argument("--tag", type=str, default="v3")
    ap.add_argument("--n-bases", type=int, default=60)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    smoke = _generate(n_bases=10, max_on_every_base=False, depth=args.depth)
    smoke_path = args.out / f"p1_searchwidth_smoke_{args.tag}.jsonl"
    write_jsonl(smoke_path, smoke)
    write_json(
        smoke_path.with_suffix(".manifest.json"),
        {"file": smoke_path.name, "records": len(smoke), "seed": SEED, "factor": "search_width", "B": B_LEVELS, "depth": K,
         "file_sha256": file_sha256(smoke_path), "generated_at": datetime.now(timezone.utc).isoformat()},
    )
    write_json(ROOT / "code" / "configs" / f"p1_searchwidth_smoke_{args.tag}.yaml", _config(smoke_path, 10, f"p1-searchwidth-smoke-{args.tag}"))

    full = _generate(n_bases=args.n_bases, max_on_every_base=False, depth=args.depth)
    # max validation: one variant per base (randomly pick one B level)
    rng = random.Random(SEED + 3)
    for b in range(args.n_bases):
        chosen_B = rng.choice(B_LEVELS)
        for it in full:
            if it["base_seed"] == SEED + b and it["branching"] == chosen_B:
                it["_probe_settings"] = ["low", "high", "max"]
                break
    full_path = args.out / f"p1_searchwidth_{args.tag}.jsonl"
    write_jsonl(full_path, full)
    write_json(
        full_path.with_suffix(".manifest.json"),
        {"file": full_path.name, "records": len(full), "seed": SEED, "factor": "search_width", "B": B_LEVELS, "depth": K,
         "max_validation_variants": sum(1 for it in full if "max" in it["_probe_settings"]),
         "file_sha256": file_sha256(full_path), "generated_at": datetime.now(timezone.utc).isoformat()},
    )
    write_json(ROOT / "code" / "configs" / f"p1_searchwidth_{args.tag}.yaml", _config(full_path, 60, f"p1-searchwidth-{args.tag}"))
    print(f"smoke: {len(smoke)} items -> {smoke_path}")
    print(f"full: {len(full)} items -> {full_path}")
    print("saved configs/p1_searchwidth_smoke.yaml + p1_searchwidth.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
