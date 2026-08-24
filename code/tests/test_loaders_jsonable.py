"""Regression tests for JSON-safe conversion of numpy/pandas values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reasoning_efficiency.loaders.base import make_jsonable


def test_numpy_1d_array_becomes_list():
    arr = np.array(["House", "Name", "Color"])
    out = make_jsonable(arr)
    assert out == ["House", "Name", "Color"]
    assert isinstance(out, list)


def test_numpy_2d_array_becomes_nested_list():
    arr = np.array([["1", "Bob", "red"], ["2", "Alice", "blue"]], dtype=object)
    out = make_jsonable(arr)
    assert out == [["1", "Bob", "red"], ["2", "Alice", "blue"]]


def test_nested_object_array_inside_dict():
    data = {
        "header": np.array(["House", "Name"]),
        "rows": np.array([np.array(["1", "Bob"]), np.array(["2", "Alice"])], dtype=object),
    }
    out = make_jsonable(data)
    assert out["header"] == ["House", "Name"]
    assert out["rows"] == [["1", "Bob"], ["2", "Alice"]]


def test_pandas_series_becomes_list():
    s = pd.Series(["a", "b"])
    assert make_jsonable(s) == ["a", "b"]


def test_json_roundtrip_of_solution():
    import json

    solution = {
        "header": np.array(["House", "Name"]),
        "rows": np.array([np.array(["1", "Bob"]), np.array(["2", "Alice"])], dtype=object),
    }
    dumped = json.dumps(make_jsonable(solution))
    loaded = json.loads(dumped)
    assert loaded == {"header": ["House", "Name"], "rows": [["1", "Bob"], ["2", "Alice"]]}
