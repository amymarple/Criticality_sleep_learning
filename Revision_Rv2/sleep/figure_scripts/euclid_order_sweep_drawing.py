"""Drawing functions from analysis_scripts/revision/ (euclid_order_sweep.py) used by the
figure notebooks.
"""
from math import comb
import numpy as np




# --- euclid_order_sweep.py
def _manifold_basis(n, b=2):
    """Columns span the beta manifold. Exact port of calc_dbv1.m's `x`."""
    m = b // 2 - 1
    x = np.zeros((n, n - m))
    for k in range(m + 1, n + 1):
        for t in range(1, k + 1):
            x[t - 1, k - m - 1] = comb(k, t) * (-1) ** (t + 1)
    return x


# --- euclid_order_sweep.py
_BASIS = {}


# --- euclid_order_sweep.py
def euclid_db(kern, b=2):
    """d2 (2024 Euclidean): L2 distance from kern to the beta manifold.

    Port of prox_crit_toolkit/src/calc_dbv1.m (QR least squares -> lstsq).
    """
    kern = np.asarray(kern, float)
    n = kern.size
    key = (n, b)
    if key not in _BASIS:
        x = _manifold_basis(n, b)
        X = x - x[:, [0]]
        _BASIS[key] = (x[:, 0].copy(), X[:, 1:].copy())
    x0, X = _BASIS[key]
    v, *_ = np.linalg.lstsq(X, kern - x0, rcond=None)
    return float(np.sqrt(np.sum((kern - (x0 + X @ v)) ** 2)))
