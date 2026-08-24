"""Data-integrity tests over the real downloaded/processed pilot datasets."""

from __future__ import annotations

from pathlib import Path

import pytest

from reasoning_efficiency.eval import evaluate_answer
from reasoning_efficiency.io import load_yaml, read_json, read_jsonl
from reasoning_efficiency.schema import validate_record

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "code" / "configs" / "datasets.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return load_yaml(CONFIG_PATH)


def _processed_path(key: str) -> Path:
    return ROOT / "datasets" / "processed" / f"{key}.jsonl"


@pytest.mark.parametrize("key", ["math500", "zebralogic", "easy2hard", "aime", "gpqa", "livecodebench"])
def test_processed_file_exists_and_valid(cfg, key):
    path = _processed_path(key)
    assert path.exists(), f"processed file missing: {path} (run prepare_datasets.py)"
    records = read_jsonl(path)
    assert len(records) > 0, f"{key}: empty processed file"

    dataset_name = cfg["datasets"][key]["name"]
    errors = [e for rec in records for e in validate_record(rec, dataset=dataset_name)]
    assert not errors, f"{key}: {len(errors)} schema errors; first: {errors[:5]}"

    ids = [rec["id"] for rec in records]
    assert len(set(ids)) == len(ids), f"{key}: duplicate ids"


@pytest.mark.parametrize("key", ["math500", "zebralogic", "easy2hard", "aime", "gpqa", "livecodebench"])
def test_expected_rows(cfg, key):
    expected = cfg["datasets"][key].get("expected_rows")
    if expected is not None:
        assert len(read_jsonl(_processed_path(key))) == expected


@pytest.mark.parametrize("key", ["math500", "zebralogic", "easy2hard", "aime", "gpqa", "livecodebench"])
def test_raw_manifest_matches_config(cfg, key):
    manifest = read_json(ROOT / "datasets" / "raw" / key / "manifest.json")
    d = cfg["datasets"][key]
    assert manifest["repo_id"] == d["repo_id"]
    assert manifest["revision"] == d["revision"]
    manifest_files = {f["path"] for f in manifest["files"]}
    expected = list(d["files"]) + ["README.md"]
    missing = [rel for rel in expected if rel not in manifest_files]
    assert not missing, f"{key}: files missing from manifest: {missing}"
    for entry in manifest["files"]:
        assert entry["size"] > 0, f"{key}: zero-size file in manifest: {entry['path']}"
        assert (ROOT / "datasets" / "raw" / key / entry["path"]).exists(), (
            f"{key}: manifest entry missing on disk: {entry['path']}"
        )


def _supported_evaluable(record: dict) -> bool:
    if record["dataset"] in ("AIME", "GPQA-Diamond", "MATH-500"):
        return True
    if record["dataset"] == "LiveCodeBench":
        return False  # code execution is covered by unit tests (self-check would run JSON as code)
    if record["dataset"] == "ZebraLogicBench":
        return record.get("metadata", {}).get("mode") in ("mc", "grid")
    if record["dataset"] == "Easy2Hard-Bench":
        return record.get("metadata", {}).get("subset") in ("E2H-AMC", "E2H-ARC", "E2H-Winogrande", "E2H-GSM8K")
    return False


@pytest.mark.parametrize("key", ["math500", "zebralogic", "easy2hard", "aime", "gpqa"])
def test_evaluator_self_consistency(key):
    records = read_jsonl(_processed_path(key))
    checked = 0
    for rec in records[:50]:
        if not _supported_evaluable(rec):
            continue
        assert evaluate_answer(rec, rec["answer"]) is True, (
            f"{key} id={rec['id']}: ground truth failed self-check: {rec['answer']!r}"
        )
        checked += 1
    assert checked > 0, f"{key}: no evaluable records in first 50 (loader/evaluator mismatch)"


def test_processed_manifests_exist():
    for key in ("math500", "zebralogic", "easy2hard"):
        path = ROOT / "datasets" / "processed" / f"{key}_manifest.json"
        assert path.exists(), f"processed manifest missing: {path}"
        manifest = read_json(path)
        assert manifest["schema_version"] == "v0.1"
        assert manifest["records"] > 0
