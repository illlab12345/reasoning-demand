#!/usr/bin/env python
"""Run all unit and data-integrity tests."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "code" / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "code" / "src"))

import pytest  # noqa: E402

raise SystemExit(pytest.main([str(ROOT / "code" / "tests"), "-v", "-ra"]))

