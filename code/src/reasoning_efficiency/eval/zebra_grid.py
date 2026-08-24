"""ZebraLogic grid_mode deterministic evaluator.

Model output is parsed into a canonical assignment
``{(entity, attribute): value}`` and compared cell-by-cell with the ground
truth. Parsing is order-insensitive; values are normalized (trim, unicode
quotes, casefold) but NOT semantically relaxed.

Accepted v1 output formats:
1. Markdown table (first column = house/entity, header = attributes).
2. JSON object {entity: {attribute: value}} or list of such rows.
3. Plain lines "Entity: attribute = value, ...".
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


def _norm(s: Any) -> str:
    s = str(s)
    # decorative curly quotes are stripped; straight apostrophes are kept
    for ch in ("“", "”", "‘", "’", "`", "´"):
        s = s.replace(ch, "")
    s = re.sub(r"\s+", " ", s).strip()
    return s.casefold()


def _cell_key(entity: Any, attribute: Any) -> tuple[str, str]:
    return (_norm(entity), _norm(attribute))


def parse_ground_truth(record: dict[str, Any]) -> dict[tuple[str, str], str]:
    """Build canonical assignment from a processed ZebraLogic grid record."""
    metadata = record.get("metadata", {})
    solution = metadata.get("solution")
    if not solution:
        raw = record.get("answer", "")
        try:
            solution = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as e:
            raise ValueError(f"cannot parse ground truth solution: {e}") from e
    header = [str(h) for h in solution.get("header", [])]
    rows = solution.get("rows", [])
    if not header or not rows:
        raise ValueError("ground truth solution missing header/rows")
    out: dict[tuple[str, str], str] = {}
    for row in rows:
        if len(row) < len(header):
            continue
        entity = row[0]
        for attr, value in zip(header[1:], row[1:]):
            out[_cell_key(entity, attr)] = _norm(value)
    return out


def _parse_markdown(text: str) -> dict[tuple[str, str], str] | None:
    lines = [ln.strip() for ln in text.splitlines()]
    table = [ln for ln in lines if ln.startswith("|")]
    if len(table) < 2:
        return None
    header = [c.strip() for c in table[0].strip("|").split("|")]
    start = 1
    if len(table) > 1 and re.fullmatch(r"[\s|:\-]+", table[1]):
        start = 2
    out: dict[tuple[str, str], str] = {}
    for line in table[start:]:
        row = [c.strip() for c in line.strip("|").split("|")]
        if len(row) < len(header) or not row[0]:
            continue
        entity = row[0]
        for attr, value in zip(header[1:], row[1:]):
            if not value or value in ("___", "-"):
                continue
            out[_cell_key(entity, attr)] = _norm(value)
    return out or None


def _parse_json(text: str) -> dict[tuple[str, str], str] | None:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    out: dict[tuple[str, str], str] = {}
    if isinstance(data, dict) and "header" in data and "rows" in data:
        # solution-format JSON (same as ground truth)
        header = [str(h) for h in data["header"]]
        for row in data.get("rows", []):
            if len(row) < len(header):
                continue
            entity = row[0]
            for attr, value in zip(header[1:], row[1:]):
                out[_cell_key(entity, attr)] = _norm(value)
    elif isinstance(data, dict):
        for entity, attrs in data.items():
            if isinstance(attrs, dict):
                for attr, value in attrs.items():
                    out[_cell_key(entity, attr)] = _norm(value)
    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            entity = None
            for key in ("house", "House", "entity", "Entity", "row", "Row"):
                if key in item:
                    entity = item[key]
                    break
            if entity is None:
                continue
            for attr, value in item.items():
                if attr in ("house", "House", "entity", "Entity", "row", "Row"):
                    continue
                out[_cell_key(entity, attr)] = _norm(value)
    return out or None


def _parse_text_lines(text: str) -> dict[tuple[str, str], str] | None:
    out: dict[tuple[str, str], str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([^:]+):\s*(.+)$", line)
        if not m:
            continue
        entity = m.group(1).strip()
        body = m.group(2)
        for part in body.split(","):
            pm = re.match(r"^\s*([^=:]+?)\s*[=:]\s*(.+?)\s*$", part)
            if pm:
                out[_cell_key(entity, pm.group(1))] = _norm(pm.group(2))
    return out or None


def parse_model_output(text: str) -> dict[tuple[str, str], str]:
    """Parse a model grid output into canonical assignment; raises ValueError."""
    for parser in (_parse_markdown, _parse_json, _parse_text_lines):
        result = parser(text)
        if result is not None:
            return result
    raise ValueError("unparseable grid output (expected Markdown table, JSON, or Entity: attr=value lines)")


@dataclass
class GridEvalResult:
    correct: bool
    cell_accuracy: float
    n_cells: int
    n_matched: int
    parse_error: str | None = None
    missing_cells: int = 0
    extra_cells: int = 0
    details: dict[str, Any] = field(default_factory=dict)


def evaluate_grid(record: dict[str, Any], response: str) -> GridEvalResult:
    """Evaluate a model grid response against a processed record."""
    ground_truth = parse_ground_truth(record)
    try:
        predicted = parse_model_output(response)
    except ValueError as e:
        return GridEvalResult(
            correct=False,
            cell_accuracy=0.0,
            n_cells=len(ground_truth),
            n_matched=0,
            parse_error=str(e),
            missing_cells=len(ground_truth),
            extra_cells=0,
        )

    gt_keys = set(ground_truth)
    pred_keys = set(predicted)
    matched = sum(1 for k in gt_keys if k in predicted and predicted[k] == ground_truth[k])
    n_cells = len(gt_keys)
    correct = pred_keys == gt_keys and matched == n_cells
    return GridEvalResult(
        correct=correct,
        cell_accuracy=matched / n_cells if n_cells else 0.0,
        n_cells=n_cells,
        n_matched=matched,
        missing_cells=len(gt_keys - pred_keys),
        extra_cells=len(pred_keys - gt_keys),
        details={
            "n_predicted_cells": len(pred_keys),
            "wrong_cells": sorted(gt_keys - {k for k in gt_keys if k in predicted and predicted[k] == ground_truth[k]})[:20],
        },
    )
