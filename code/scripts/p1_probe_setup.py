#!/usr/bin/env python
"""P1 probe setup (zero API calls):

1. Mechanism probe: 30 synthetic multi-hop arithmetic items (10 per depth
   2/4/6), generated deterministically (seed 42).
2. Prospective smoke: 30 untouched items from development pools
   (MATH-500 10, E2H-AMC 10, ZebraLogic grid 5, LiveCodeBench 5), selected
   deterministically, with the frozen stratum-MSRB router rule applied.

Outputs:
  datasets/probe/mechanism_probe_v1.jsonl
  datasets/probe/prospective_smoke_v1.jsonl
  configs/p1_probe.yaml (frozen router rule + probe settings)
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

from reasoning_efficiency.io import file_sha256, load_yaml, read_jsonl, write_json, write_jsonl  # noqa: E402

SEED = 42
DEPTHS = [8, 16, 24]
PER_DEPTH = 10


def _make_mechanism_items() -> list[dict]:
    rng = random.Random(SEED)
    items = []
    for depth in DEPTHS:
        for i in range(PER_DEPTH):
            start = rng.randint(10000, 99999)
            steps = []
            v = start
            for _ in range(depth):
                op = rng.choice(["+", "*"])
                k = rng.randint(11, 99) if op == "+" else rng.randint(2, 5)
                steps.append((op, k))
                if op == "+":
                    v += k
                else:
                    v *= k
            chain = " -> ".join(f"{op} {k}" for op, k in steps)
            question = (
                f"Start with the number {start}. Apply the following operations in order: {chain}. "
                f"What is the final value? Provide the final answer in a line starting with \"Answer:\"."
            )
            items.append(
                {
                    "id": f"mech_d{depth}_{i:02d}",
                    "dataset": "MechanismProbe",
                    "domain": "math",
                    "stratum": f"depth{depth}",
                    "question": question,
                    "answer": str(v),
                    "difficulty": None,
                    "metadata": {"depth": depth, "start": start, "steps": steps},
                    "_probe_settings": ["low", "high"],
                    "_prompt": "aime_v1",
                }
            )
    return items


def _untouched(processed: Path, dataset_key: str, pilot_file: Path) -> list[dict]:
    all_items = read_jsonl(processed / f"{dataset_key}.jsonl")
    used = {r["id"] for r in read_jsonl(pilot_file)}
    return [r for r in all_items if r["id"] not in used]


def _pick(items: list[dict], per_stratum: dict[str, int], rng: random.Random, stratum_of) -> list[dict]:
    chosen = []
    for st, n in per_stratum.items():
        pool = [r for r in items if stratum_of(r) == st]
        idx = sorted(rng.sample(range(len(pool)), min(n, len(pool))))
        chosen.extend(pool[i] for i in idx)
    return chosen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=ROOT / "datasets" / "processed")
    ap.add_argument("--pilot", type=Path, default=ROOT / "datasets" / "pilot")
    ap.add_argument("--out", type=Path, default=ROOT / "datasets" / "probe")
    ap.add_argument("--complexity", type=Path, default=ROOT / "data" / "tables" / "zebra_complexity.csv")
    ap.add_argument("--paper-tables", type=Path, default=ROOT / "data" / "tables" / "paper_tables.json")
    ap.add_argument("--rule", type=Path, default=None, help="frozen router rule JSON (overrides paper_tables derivation)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED + 1)

    # mechanism items
    mech = _make_mechanism_items()
    mech_path = args.out / "mechanism_probe_v1.jsonl"
    write_jsonl(mech_path, mech)
    write_json(
        mech_path.with_suffix(".manifest.json"),
        {
            "file": mech_path.name,
            "records": len(mech),
            "seed": SEED,
            "method": "synthetic multi-hop arithmetic, 10 per depth 2/4/6",
            "strata": {f"depth{d}": PER_DEPTH for d in DEPTHS},
            "file_sha256": file_sha256(mech_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"mechanism: {len(mech)} items -> {mech_path}")

    # frozen router rule: from --rule JSON, or derived from development stratum MSRB (paper_tables.json)
    if args.rule:
        data = json.loads(args.rule.read_text(encoding="utf-8"))
        frozen_rule = data["rule"] if isinstance(data, dict) and isinstance(data.get("rule"), dict) else data
    else:
        dev = json.loads(args.paper_tables.read_text(encoding="utf-8"))["stratum"]
        frozen_rule = {f"{ds}|{st}": v["msrb"] for ds, strata in dev.items() for st, v in strata.items()}

    # prospective smoke: untouched items
    math_pool = _untouched(args.processed, "math500", args.pilot / "math500_pilot_v1.jsonl")
    amc_pool = _untouched(args.processed, "easy2hard", args.pilot / "easy2hard_amc_pilot_v1.jsonl")
    zebra_pool = _untouched(args.processed, "zebralogic", args.pilot / "zebralogic_grid_pilot_v1.jsonl")
    lcb_pool = _untouched(args.processed, "livecodebench", args.pilot / "livecodebench_pilot_v1.jsonl")
    amc_pool = [r for r in amc_pool if r["metadata"].get("subset") == "E2H-AMC"]
    zebra_pool = [r for r in zebra_pool if r["metadata"].get("mode") == "grid"]

    math_sel = _pick(math_pool, {str(i): 2 for i in range(1, 6)}, rng, lambda r: str(r["metadata"]["level"]))
    comp = pd.read_csv(args.complexity)
    comp_map = dict(zip(comp["id"], comp["quintile"]))
    zebra_sel = _pick(zebra_pool, {str(i): 1 for i in range(1, 6)}, rng, lambda r: str(comp_map.get(r["id"], "1")))
    lcb_sel = _pick(lcb_pool, {"easy": 2, "medium": 2, "hard": 1}, rng, lambda r: r["metadata"].get("difficulty", "easy"))

    # AMC stratum: recompute quintile with the same rank method used for the dev sample
    if amc_pool:
        diffs = [r["difficulty"] for r in amc_pool]
        rank = pd.Series(diffs).rank(method="first", ascending=True).astype(int)
        n = len(amc_pool)
        bin_size = -(-n // 5)
        quintile = ((rank - 1) // bin_size).clip(0, 4) + 1
        for r, q in zip(amc_pool, quintile):
            r["_quintile"] = str(int(q))
        amc_sel = _pick(amc_pool, {str(i): 2 for i in range(1, 6)}, rng, lambda r: r["_quintile"])

    prospective = []
    for src, sel, prompt, dkey in (
        ("math500", math_sel, "math_v1", "math500"),
        ("easy2hard_amc", amc_sel, "e2h_amc_v1", "easy2hard_amc"),
        ("zebralogic_grid", zebra_sel, "zebra_grid_v1", "zebralogic_grid"),
        ("livecodebench", lcb_sel, "livecodebench_v1", "livecodebench"),
    ):
        for r in sel:
            st = r.get("_quintile") or r.get("_stratum") or (str(comp_map.get(r["id"], "1")) if src == "zebralogic_grid" else "1")
            if src == "math500":
                st = str(r["metadata"]["level"])
            if src == "livecodebench":
                st = r["metadata"]["difficulty"]
            router_setting = frozen_rule.get(f"{dkey}|{st}", "high")
            rec = {
                "id": f"prosp_{src}_{r['id']}",
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
            prospective.append(rec)

    prosp_path = args.out / "prospective_smoke_v1.jsonl"
    write_jsonl(prosp_path, prospective)
    write_json(
        prosp_path.with_suffix(".manifest.json"),
        {
            "file": prosp_path.name,
            "records": len(prospective),
            "seed": SEED + 1,
            "method": "untouched development-pool items; frozen stratum-MSRB router",
            "strata": {f"{r['id'].split('_')[1]}|{r['stratum']}": 1 for r in prospective},
            "router_settings": {s: sum(1 for r in prospective if r["_router_setting"] == s) for s in ("low", "high", "max")},
            "file_sha256": file_sha256(prosp_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(f"prospective: {len(prospective)} items -> {prosp_path}")
    print("frozen router rule:", frozen_rule)

    cfg = {
        "schema_version": "p1-probe-v3",
        "model": "deepseek-v4-flash",
        "repeats": 3,
        "temperature": 0.0,
        "seed": None,
        "frozen_router_rule": frozen_rule,
        "mechanism_items": str(mech_path),
        "prospective_items": str(prosp_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(ROOT / "code" / "configs" / "p1_probe.yaml", cfg)
    print("saved configs/p1_probe.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
