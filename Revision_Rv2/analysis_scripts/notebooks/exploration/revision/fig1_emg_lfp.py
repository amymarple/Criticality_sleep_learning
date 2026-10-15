"""
fig1_emg_lfp.py -- LFP-derived EMG per 30 s window, for the state-space UMAP.

The window features already hold the slow-wave score (`sw_broadband`) and theta
(`theta_ratio`, fig1_theta_ratio.py); EMG (muscle tone) is the missing scoring axis.

buzcode SleepScoreMaster's own EMG metric, stored beside `thratio` in
<basename>.SleepState.states.mat -> detectorinfo.detectionparms.SleepScoreMetrics.EMG, on the
same 1 s `t_clus` grid. Definition (neurocode/brainStates/SleepScoreMaster/ClusterStates_GetMetrics.m,
lines 220-232): EMGFromLFP (high-frequency correlation across channels), smoothed, sampled at
t_clus, min-max normalised to 0..1 within the session. Window value = mean of the t_clus
samples inside the window.

SleepScoreMaster uses this metric to separate wake from sleep.

    -> data/revision/log/fig1_emg_lfp_windows.csv   (the UMAP's windows: session, state, center_s)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id       # noqa: E402
from fig1_metrics import WINDOW_S                                                # noqa: E402
from fig1_dropin_metrics import _find_file, _loadmat_any                         # noqa: E402
from fig1_theta_ratio import theta_ratio_window as metric_window                 # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOG = os.path.join(REPO, "data", "revision", "log")
OUT = os.path.join(LOG, "fig1_emg_lfp_windows.csv")


def load_emg(basepath):
    """(t_clus, EMG) on buzcode's 1 s grid, sorted, NaNs dropped."""
    path = _find_file(basepath, ".SleepState.states.mat")
    if path is None:
        raise FileNotFoundError(f"no SleepState.states.mat in {basepath}")
    msm = _loadmat_any(path)["SleepState"]["detectorinfo"]["detectionparms"]["SleepScoreMetrics"]
    t = np.asarray(msm["t_clus"], float).ravel()
    e = np.asarray(msm["EMG"], float).ravel()
    if t.size != e.size:
        raise ValueError(f"t_clus/EMG length mismatch in {path}")
    ok = np.isfinite(t) & np.isfinite(e)
    o = np.argsort(t[ok])
    return t[ok][o], e[ok][o]


def main():
    F = pd.read_csv(os.path.join(LOG, "fig1_state_space_features.csv"), usecols=["session", "state", "center_s"])
    rows = []
    for bp in FIG1_SESSIONS:
        a, d = session_id(bp)
        sess = f"{a}_{d}_CA1"
        W = F[F.session == sess]
        if W.empty:
            continue
        t, e = load_emg(bp)
        for r in W.itertuples(index=False):
            v, n = metric_window(t, e, r.center_s - WINDOW_S / 2, r.center_s + WINDOW_S / 2)
            rows.append(dict(session=sess, state=r.state, center_s=r.center_s, EMG_lfp=v, emg_n=n))
        g = pd.DataFrame(rows[-len(W):])
        print(f"{sess}: {len(W)} windows, grid {np.median(np.diff(t)):.2f} s, "
              + ", ".join(f"{s} {g[g.state == s].EMG_lfp.median():.2f}" for s in ("THETA", "nonTHETA", "REM", "NREM")),
              flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(assert_local_path(OUT), index=False)
    print(f"-> {OUT}: {len(D)} windows, {D.EMG_lfp.notna().mean():.1%} with a value")
    print(D.groupby("state").EMG_lfp.describe()[["count", "25%", "50%", "75%"]].round(3).to_string())


if __name__ == "__main__":
    main()
