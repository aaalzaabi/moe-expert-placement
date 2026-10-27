"""Shared fixtures for the unittest suite (stdlib only; pytest can also run it)."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from moeplace import config as C  # noqa: E402
from moeplace import data as D  # noqa: E402
from moeplace import policy as POL  # noqa: E402


@lru_cache(maxsize=None)
def routing() -> pd.DataFrame:
    return D.load_routing()


@lru_cache(maxsize=None)
def default_layouts():
    return POL.build_layouts(routing(), C.DEFAULT)[1]


def assert_frames_close(tc, actual: pd.DataFrame, expected: pd.DataFrame, cols, label=""):
    actual, expected = actual.reset_index(drop=True), expected.reset_index(drop=True)
    tc.assertEqual(len(actual), len(expected), f"{label}: row count")
    for c in cols:
        a, e = actual[c], expected[c]
        if pd.api.types.is_numeric_dtype(e) or pd.api.types.is_numeric_dtype(a):
            ok = np.allclose(a.astype(float), e.astype(float), rtol=1e-9, atol=1e-9, equal_nan=True)
        else:
            ok = (a.astype(str) == e.astype(str)).all()
        tc.assertTrue(ok, f"{label}: column {c!r} differs\nactual={a.tolist()[:6]}\nexpected={e.tolist()[:6]}")
