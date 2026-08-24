from __future__ import annotations

from pathlib import Path

from reasoning_efficiency.io import read_jsonl

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "datasets" / "pilot"


def _load(name: str):
    return read_jsonl(PILOT / f"{name}.jsonl")


def _processed_ids(dataset: str, mode: str | None = None, subset: str | None = None) -> set[str]:
    rows = read_jsonl(ROOT / "datasets" / "processed" / f"{dataset}.jsonl")
    if mode is not None:
        rows = [r for r in rows if r["metadata"].get("mode") == mode]
    if subset is not None:
        rows = [r for r in rows if r["metadata"].get("subset") == subset]
    return {r["id"] for r in rows}


def test_math500_pilot_strata():
    pilot = _load("math500_pilot_v1")
    assert len(pilot) == 100
    strata = {}
    for r in pilot:
        strata[r["_stratum"]] = strata.get(r["_stratum"], 0) + 1
    assert strata == {"1": 20, "2": 20, "3": 20, "4": 20, "5": 20}
    ids = {r["id"] for r in pilot}
    assert ids <= _processed_ids("math500")


def test_zebra_grid_pilot_strata():
    pilot = _load("zebralogic_grid_pilot_v1")
    assert len(pilot) == 100
    strata = {}
    for r in pilot:
        strata[r["_stratum"]] = strata.get(r["_stratum"], 0) + 1
    assert strata == {"1": 20, "2": 20, "3": 20, "4": 20, "5": 20}
    ids = {r["id"] for r in pilot}
    assert ids <= _processed_ids("zebralogic", mode="grid")


def test_easy2hard_amc_pilot_strata():
    pilot = _load("easy2hard_amc_pilot_v1")
    assert len(pilot) == 100
    strata = {}
    for r in pilot:
        strata[r["_stratum"]] = strata.get(r["_stratum"], 0) + 1
    assert strata == {"1": 20, "2": 20, "3": 20, "4": 20, "5": 20}
    ids = {r["id"] for r in pilot}
    assert ids <= _processed_ids("easy2hard", subset="E2H-AMC")


def test_smoke_and_calibration_nested():
    for base in ("math500", "zebralogic_grid", "easy2hard_amc"):
        pilot = {r["id"] for r in _load(f"{base}_pilot_v1")}
        cal = {r["id"] for r in _load(f"{base}_calibration_v1")}
        smoke = {r["id"] for r in _load(f"{base}_smoke_v1")}
        assert len(smoke) == 10
        assert len(cal) == 30
        assert smoke <= cal <= pilot


def test_smoke_strata_balance():
    for base in ("math500", "zebralogic_grid", "easy2hard_amc"):
        smoke = _load(f"{base}_smoke_v1")
        strata = {}
        for r in smoke:
            strata[r["_stratum"]] = strata.get(r["_stratum"], 0) + 1
        assert strata == {"1": 2, "2": 2, "3": 2, "4": 2, "5": 2}
