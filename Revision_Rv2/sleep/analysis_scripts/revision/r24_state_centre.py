"""
r24_state_centre.py -- state-resolved d2 (2024 Euclidean) and branching ratio m with the wake split of Fig 1D -- a
window is THETA if its CENTRE lies in SleepState THETA -- on non-overlapping windows.

The Fig-1 tiles (fig1_session_medians.csv) label a wake window THETA / nonTHETA only if it lies wholly inside
WAKE & THETA or WAKE - THETA, so windows across a theta boundary do not exist there. Fig 1D
(`Fig1_D_histogram_S1_E.ipynb`, `split_wake_into_theta_nontheta`) labels every WAKE window by its centre; the two
labellings are compared in d2_wake_env_audit.py.

DESIGN. Windows = the MATLAB producer's `_error_bar.mat` 30 s windows (the d2 of the .mat files),
thinned to centres >= 30 s apart within each state (`d2_err_60s_vs_30s.thin`), so none overlap.
States NREM / REM as in the .mat; WAKE -> THETA if the centre lies in SleepState THETA (raw
intervals, as `load_theta_epochs`), else nonTHETA. m on the same [centre - 15, centre + 15] s:
MR estimator (fit_m), active-unit counts at 4 ms, k_max = 200, pyramidal and all CA1
units (the population d2 uses), units with rate > 0.2 Hz as the Fig-1 pipeline; tau = -4 ms / ln m.
Per session and state: medians over windows -> the same columns as fig1_session_medians.csv.
The windows table is the input of d2_arms_raw.py (LABEL=centre).

Reads the share read-only; writes only to D:.
    -> data/revision/log/r24_state_centre_windows.csv
    -> data/revision/log/fig1_session_medians_centre.csv
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path, FIG1_SESSIONS, session_id   # noqa: E402
from fig1_metrics import active_unit_counts, fit_m, BIN_BR_S, MIN_FIRING_RATE  # noqa: E402
from state_level_br import rk_curve_multi                                    # noqa: E402
from d2_err_60s_vs_30s import read as read_mat, thin                         # noqa: E402
from d2_wake_env_audit import in_intervals                                   # noqa: E402

LOGDIR = os.path.abspath(os.path.join(_HERE, "..", "data", "revision", "log"))
OUT_WIN = os.path.join(LOGDIR, "r24_state_centre_windows.csv")
OUT_MED = os.path.join(LOGDIR, "fig1_session_medians_centre.csv")
K_MAX, HALF = 200, 15.0


def m_fit(units, t0, t1):
    A = active_unit_counts(units, t0, t1, BIN_BR_S)
    K, r_k, se, n = rk_curve_multi([A], K_MAX + 1)
    return fit_m(K, r_k, se, K_MAX)


def session(bp):
    from neuro_py.io import loading
    a, d = session_id(bp); sid = f"{a}_{d}"
    M = read_mat(bp, "_error_bar")
    if M is None:
        return pd.DataFrame()
    M = thin(M.dropna(subset=["d2"]), 30).sort_values(["state", "center"]).copy()
    th = loading.load_SleepState_states(bp).get("THETA", np.zeros((0, 2)))
    wake = M.state == "WAKE"
    M.loc[wake, "state"] = np.where(in_intervals(M.loc[wake, "center"].to_numpy(float), th), "THETA", "nonTHETA")
    st, cm = loading.load_spikes(bp, brainRegion="CA1")
    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    ctype = np.asarray(cm.putativeCellType)
    units = [np.sort(np.asarray(u, float)) for u, k in zip(np.asarray(st.data, dtype=object), keep) if k]
    pyr = [u for u, c in zip(units, ctype[keep]) if "Pyramidal" in str(c)]
    M["m_pyr"] = [m_fit(pyr, c - HALF, c + HALF) for c in M.center]
    M["m_all"] = [m_fit(units, c - HALF, c + HALF) for c in M.center]
    M["session"] = sid
    print(f"  {sid:12s} " + ", ".join(f"{s} {n}" for s, n in M.state.value_counts().items()), flush=True)
    return M[["session", "state", "center", "d2", "err", "m_pyr", "m_all"]]


def main():
    from multiprocessing import Pool
    with Pool(int(os.environ.get("NPROC", 6))) as pool:
        parts = pool.map(session, FIG1_SESSIONS)
    W = pd.concat(parts, ignore_index=True)
    W.to_csv(assert_local_path(OUT_WIN), index=False)
    tau = lambda m: np.where((m > 0) & (m < 1), -1000 * BIN_BR_S / np.log(m), np.nan)
    W["tau_pyr"], W["tau_all"] = tau(W.m_pyr.to_numpy(float)), tau(W.m_all.to_numpy(float))
    S = W.groupby(["session", "state"]).agg(
        d2=("d2", "median"), BR_pyr_k200=("m_pyr", "median"), BR_all_k200=("m_all", "median"),
        tau_ms_pyr_k200=("tau_pyr", "median"), tau_ms_all_k200=("tau_all", "median"),
        n_windows=("d2", "size")).reset_index()
    S.to_csv(assert_local_path(OUT_MED), index=False)
    print(f"\n{S.session.nunique()} sessions, {len(W)} non-overlapping windows -> {OUT_MED}")
    print(S.groupby("state")[["d2", "BR_pyr_k200", "BR_all_k200", "n_windows"]].median().round(4))


if __name__ == "__main__":
    main()
