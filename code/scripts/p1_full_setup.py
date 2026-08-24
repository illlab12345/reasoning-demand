#!/usr/bin/env python
"""Full P1 locked setup (zero API calls).

Prospective (N=300): untouched items from math500 (100), E2H-AMC (100),
ZebraLogic grid (50), LiveCodeBench (50), excluding the 30 smoke items;
frozen Router v3 (token-cost-aware) vs Always High, 3 reps; `max` on a 20%
stratified subsample (60 items) for ART-style analysis.

Mechanism (3 factors x 3 levels x 20 items = 180): depth (8/16/24 steps),
distractor load (0/2/4 irrelevant numbers), constraint count (4/8/12 clues in
a logic chain). Settings [low, high], 3 reps; `max` on 30 validation items.

Outputs:
  datasets/probe/p1_full_prospective_v1.jsonl
  datasets/probe/p1_full_mechanism_v1.jsonl
  configs/p1_full.yaml
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

import pandas as pd  # noqa: E402

from reasoning_efficiency.io import file_sha256, read_jsonl, write_json, write_jsonl  # noqa: E402

SEED = 20260817
EPSILON = 0.03


def _chain_item(rng: random.Random, depth: int, distractor_level: int, factor: str) -> dict:
    start = rng.randint(10000, 99999)
    steps = []
    v = start
    for _ in range(depth):
        op = rng.choice(["+", "*"])
        k = rng.randint(11, 99) if op == "+" else rng.randint(2, 5)
        steps.append((op, k))
        v = v + k if op == "+" else v * k
    chain = " -> ".join(f"{op} {k}" for op, k in steps)
    distractors = [rng.randint(100, 9999) for _ in range(distractor_level)]
    note = ""
    if distractors:
        note = f" Note: the numbers {', '.join(map(str, distractors))} are NOT used in the computation."
    question = (
        f"Start with the number {start}. Apply the following operations in order: {chain}.{note} "
        f"What is the final value? Provide the final answer in a line starting with \"Answer:\"."
    )
    return {
        "id": f"mech_depth{depth}_dist{distractor_level}_{rng.randint(0, 999999):06d}",
        "dataset": "MechanismProbe",
        "domain": "math",
        "stratum": f"depth{depth}|dist{distractor_level}",
        "question": question,
        "answer": str(v),
        "difficulty": None,
        "metadata": {
            "factor": factor,
            "depth": depth,
            "start": start,
            "steps": steps,
            "distractors": distractors,
            "evaluator": "int",
        },
        "_probe_settings": ["low", "high"],
        "_prompt": "aime_v1",
    }


def _logic_item(rng: random.Random, n_clues: int) -> dict:
    names = ["Alice", "Bob", "Carol", "Diana", "Eve", "Frank"]
    pets = ["cat", "dog", "bird", "fish", "hamster", "rabbit"]
    drinks = ["tea", "coffee", "milk", "juice", "water", "soda"]
    houses = list(range(1, 7))
    rng.shuffle(names)
    name_house = {n: h for n, h in zip(names, houses)}
    rng.shuffle(pets)
    pet_house = {p: h for p, h in zip(pets, houses)}
    rng.shuffle(drinks)
    drink_house = {d: h for d, h in zip(drinks, houses)}
    target_name = rng.choice(names)
    target_house = name_house[target_name]
    target_pet = next(p for p, h in pet_house.items() if h == target_house)
    target_drink = next(d for d, h in drink_house.items() if h == target_house)

    relevant = [
        f"The person who drinks {target_drink} owns the {target_pet}.",
        f"{target_name} drinks {target_drink}.",
    ]
    fillers = []
    candidates = []
    for p, h in pet_house.items():
        if p != target_pet:
            candidates.append(f"The {p} lives in house {h}.")
    for d, h in drink_house.items():
        if d != target_drink:
            candidates.append(f"The person in house {h} drinks {d}.")
    for n, h in name_house.items():
        if n != target_name:
            candidates.append(f"{n} lives in house {h}.")
    rng.shuffle(candidates)
    fillers = candidates[: max(0, n_clues - 2)]
    clues = relevant + fillers
    rng.shuffle(clues)
    clue_text = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(clues))
    question = (
        f"There are 6 houses numbered 1 to 6 from left to right. Each house has a unique name, a unique pet, "
        f"and a unique drink.\nClues:\n{clue_text}\n"
        f"Question: Which pet does {target_name} own? Provide the final answer in a line starting with \"Answer:\"."
    )
    return {
        "id": f"mech_logic_clues{n_clues}_{rng.randint(0, 999999):06d}",
        "dataset": "MechanismProbe",
        "domain": "logic",
        "stratum": f"clues{n_clues}",
        "question": question,
        "answer": target_pet,
        "difficulty": None,
        "metadata": {"factor": "constraints", "n_clues": n_clues, "target_name": target_name, "evaluator": "text"},
        "_probe_settings": ["low", "high"],
        "_prompt": "aime_v1",
    }


def _mechanism_items() -> list[dict]:
    rng = random.Random(SEED)
    items = []
    for depth in (8, 16, 24):
        for _ in range(20):
            items.append(_chain_item(rng, depth, 0, "depth"))
    for dist in (0, 2, 4):
        for _ in range(20):
            items.append(_chain_item(rng, 12, dist, "distractor"))
    for clues in (4, 8, 12):
        for _ in range(20):
            items.append(_logic_item(rng, clues))
    # max validation: one per factor level (30 items)
    for it in items[:30]:
        it["_probe_settings"] = ["low", "high", "max"]
    return items


def _prospective_items(processed: Path, pilot: Path, probe_dir: Path, rule: dict) -> list[dict]:
    smoke_ids = {r["id"] for f in ("mechanism_probe_v1.jsonl", "prospective_smoke_v1.jsonl") for r in read_jsonl(probe_dir / f)}
    comp = pd.read_csv(ROOT / "data" / "tables" / "zebra_complexity.csv")
    comp_map = dict(zip(comp["id"], comp["quintile"]))

    def untouched(key: str, pilot_file: str, smoke_prefix: str) -> list[dict]:
        all_items = read_jsonl(processed / f"{key}.jsonl")
        used = {r["id"] for r in read_jsonl(pilot / pilot_file)}
        return [r for r in all_items if r["id"] not in used and f"prosp_{smoke_prefix}_{r['id']}" not in smoke_ids]

    math_pool = untouched("math500", "math500_pilot_v1.jsonl", "math500")
    amc_pool = [r for r in untouched("easy2hard", "easy2hard_amc_pilot_v1.jsonl", "easy2hard_amc") if r["metadata"].get("subset") == "E2H-AMC"]
    zebra_pool = [r for r in untouched("zebralogic", "zebralogic_grid_pilot_v1.jsonl", "zebralogic") if r["metadata"].get("mode") == "grid"]
    lcb_pool = untouched("livecodebench", "livecodebench_pilot_v1.jsonl", "livecodebench")

    if amc_pool:
        rank = pd.Series([r["difficulty"] for r in amc_pool]).rank(method="first", ascending=True).astype(int)
        n = len(amc_pool)
        q = ((rank - 1) // (-(-n // 5))).clip(0, 4) + 1
        for r, x in zip(amc_pool, q):
            r["_quintile"] = str(int(x))

    rng = random.Random(SEED + 1)

    def pick(pool: list[dict], per: dict, stratum_of) -> list[dict]:
        out = []
        for st, cnt in per.items():
            cand = [r for r in pool if stratum_of(r) == st]
            idx = sorted(rng.sample(range(len(cand)), min(cnt, len(cand))))
            out.extend(cand[i] for i in idx)
        return out

    math_sel = pick(math_pool, {str(i): 20 for i in range(1, 6)}, lambda r: str(r["metadata"]["level"]))
    amc_sel = pick(amc_pool, {str(i): 20 for i in range(1, 6)}, lambda r: r["_quintile"])
    zebra_sel = pick(zebra_pool, {str(i): 10 for i in range(1, 6)}, lambda r: str(comp_map.get(r["id"], "1")))
    lcb_sel = pick(lcb_pool, {"easy": 17, "medium": 17, "hard": 16}, lambda r: r["metadata"]["difficulty"])

    items = []
    for src, sel, prompt, dkey in (
        ("math500", math_sel, "math_v1", "math500"),
        ("easy2hard_amc", amc_sel, "e2h_amc_v1", "easy2hard_amc"),
        ("zebralogic_grid", zebra_sel, "zebra_grid_v1", "zebralogic_grid"),
        ("livecodebench", lcb_sel, "livecodebench_v1", "livecodebench"),
    ):
        for r in sel:
            if src == "math500":
                st = str(r["metadata"]["level"])
            elif src == "easy2hard_amc":
                st = r["_quintile"]
            elif src == "zebralogic_grid":
                st = str(comp_map.get(r["id"], "1"))
            else:
                st = r["metadata"]["difficulty"]
            router_setting = rule.get(f"{dkey}|{st}", "high")
            items.append(
                {
                    "id": f"p1full_{src}_{r['id']}",
                    "dataset": r["dataset"],
                    "source_id": r["source_id"],
                    "domain": r["domain"],
                    "stratum": st,
                    "question": r["question"],
                    "answer": r["answer"],
                    "difficulty": r["difficulty"],
                    "metadata": r["metadata"],
                    "_probe_settings": [router_setting, "high"],
                    "_router_setting": router_setting,
                    "_prompt": prompt,
                }
            )
    # max subsample: 20% stratified (15 per benchmark, but zebra/lcb only 5 each -> keep 60 total)
    rng2 = random.Random(SEED + 2)
    per_bench = {"math500": 20, "easy2hard_amc": 20, "zebralogic_grid": 10, "livecodebench": 10}
    for src, cnt in per_bench.items():
        group = [it for it in items if it["id"].startswith(f"p1full_{src}_") and "max" not in it["_probe_settings"]]
        chosen = rng2.sample(group, cnt)
        for it in chosen:
            it["_probe_settings"] = sorted(set(it["_probe_settings"] + ["max"]), key=lambda s: ("low", "high", "max").index(s))
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=ROOT / "datasets" / "processed")
    ap.add_argument("--pilot", type=Path, default=ROOT / "datasets" / "pilot")
    ap.add_argument("--probe", type=Path, default=ROOT / "datasets" / "probe")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "probe")
    ap.add_argument("--rule", type=Path, default=ROOT / "data" / "tables" / "router_v3_rule.json")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rule = json.loads(args.rule.read_text(encoding="utf-8"))["rule"]
    mech = _mechanism_items()
    prosp = _prospective_items(args.processed, args.pilot, args.probe, rule)

    mech_path = args.out / "p1_full_mechanism_v1.jsonl"
    prosp_path = args.out / "p1_full_prospective_v1.jsonl"
    write_jsonl(mech_path, mech)
    write_jsonl(prosp_path, prosp)
    write_json(
        mech_path.with_suffix(".manifest.json"),
        {"file": mech_path.name, "records": len(mech), "seed": SEED, "settings": ["low", "high"] + (["max"] if any("max" in it["_probe_settings"] for it in mech) else []),
         "file_sha256": file_sha256(mech_path), "generated_at": datetime.now(timezone.utc).isoformat()},
    )
    write_json(
        prosp_path.with_suffix(".manifest.json"),
        {"file": prosp_path.name, "records": len(prosp), "seed": SEED + 1, "router": "v3", "settings": ["router", "high", "max(subsample)"],
         "file_sha256": file_sha256(prosp_path), "generated_at": datetime.now(timezone.utc).isoformat()},
    )

    cfg = {
        "schema_version": "p1-full-v1",
        "model": "deepseek-v4-flash",
        "repeats": 3,
        "temperature": 0.0,
        "seed": None,
        "frozen_router_rule": rule,
        "item_files": [str(mech_path), str(prosp_path)],
        "primary_endpoints": {
            "non_inferiority_vs_high": {"epsilon": EPSILON, "ci": "item-level bootstrap, 95% one-sided"},
            "token_reduction_vs_high": {"criterion": "total tokens lower than Always High"},
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "code" / "configs" / "p1_full.yaml", cfg)
    print(f"mechanism: {len(mech)} items -> {mech_path}")
    print(f"prospective: {len(prosp)} items -> {prosp_path}")
    print("saved configs/p1_full.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
