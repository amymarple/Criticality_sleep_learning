"""fit_replay_ridge_line, from Previous_Rv1/figure_scripts/Fig2_B_Example_replay.ipynb."""
import matplotlib as mpl
import numpy as np


def fit_replay_ridge_line(P, min_col_mass_frac=0.2):
    """
    Fit a line to the replay 'ridge' in a replay probability matrix P (pos x time).
    Returns: dict with x_fit, y_fit, m, b, r2, x_used, y_used
    """
    P = np.asarray(P)
    n_pos, n_t = P.shape

    # Ridge: for each time bin, take position of max probability
    y_ridge = np.argmax(P, axis=0).astype(float)  # pos index
    x = np.arange(n_t).astype(float)              # time index

    # Gate: keep time bins where column mass is nontrivial
    col_mass = P.sum(axis=0)
    thr = min_col_mass_frac * np.nanmax(col_mass) if np.nanmax(col_mass) > 0 else 0
    keep = col_mass >= thr

    x_used = x[keep]
    y_used = y_ridge[keep]

    if len(x_used) < 2:
        return {"m": np.nan, "b": np.nan, "r2": np.nan,
                "x_fit": None, "y_fit": None, "x_used": x_used, "y_used": y_used}

    # Fit line: y = m x + b
    m, b = np.polyfit(x_used, y_used, 1)

    # R^2 on used points
    y_hat = m * x_used + b
    ss_res = np.sum((y_used - y_hat) ** 2)
    ss_tot = np.sum((y_used - np.mean(y_used)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    x_fit = np.array([x_used.min(), x_used.max()])
    y_fit = m * x_fit + b

    return {"m": m, "b": b, "r2": r2,
            "x_fit": x_fit, "y_fit": y_fit, "x_used": x_used, "y_used": y_used}
