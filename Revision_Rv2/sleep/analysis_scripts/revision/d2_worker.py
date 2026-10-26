"""
d2_worker.py -- minimal worker module for the parallel AR order sweep (ar_order_sweep.py).

Deliberately tiny: multiprocessing on Windows uses spawn, so every child re-imports
this module. Importing `run_fig1_metrics` (and therefore neuro_py/nelpy) in a worker
would cost ~10 s of startup per process. This imports only numpy and d2_fast.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(                       # methods/d2
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))), "methods", "d2"))

import numpy as np  # noqa: E402
from d2_fast import d2_from_timeseries  # noqa: E402


def d2_one(task):
    """task = (key, x, order, dt) -> (key, d2). Never raises; NaN on failure."""
    key, x, order, dt = task
    try:
        v, _ = d2_from_timeseries(np.asarray(x, float), order=order, dt=dt)
        return key, float(v)
    except Exception:
        return key, float("nan")
