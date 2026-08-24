from __future__ import annotations

from reasoning_efficiency.schema import validate_record, validate_records


def _valid_record() -> dict:
    return {
        "id": "math500_0000",
        "source_id": "0",
        "dataset": "MATH-500",
        "domain": "math",
        "question": "What is 2+2?",
        "answer": "4",
        "difficulty": None,
        "metadata": {"subject": "algebra"},
    }


def test_valid_record_passes():
    assert validate_record(_valid_record()) == []


def test_valid_record_with_dataset_mismatch():
    rec = _valid_record()
    assert validate_record(rec, dataset="ZebraLogicBench") != []


def test_missing_field_detected():
    rec = _valid_record()
    del rec["answer"]
    errors = validate_record(rec)
    assert "missing field: answer" in errors


def test_empty_question_rejected():
    rec = _valid_record()
    rec["question"] = "   "
    assert validate_record(rec) != []


def test_difficulty_must_be_numeric_or_null():
    rec = _valid_record()
    rec["difficulty"] = "hard"
    assert validate_record(rec) != []
    rec["difficulty"] = 3
    assert validate_record(rec) == []


def test_metadata_must_be_dict():
    rec = _valid_record()
    rec["metadata"] = ["not", "a", "dict"]
    assert validate_record(rec) != []


def test_duplicate_ids_detected():
    records = [_valid_record(), _valid_record()]
    summary = validate_records(records)
    assert summary["unique_ids"] is False
    assert summary["duplicate_ids"] == ["math500_0000"]

