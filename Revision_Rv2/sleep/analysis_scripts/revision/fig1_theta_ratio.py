"""
fig1_theta_ratio.py -- theta power per 30 s window, for the state-space UMAP and the Fig 1c example.

No window-level theta measure otherwise exists in the revision pipeline: the UMAP's LFP features are
the 20-45 Hz aperiodic slope and buzcode's slow-wave score (`sw_broadband`), neither of which sees theta.

buzcode SleepScoreMaster's own theta metric, stored beside `broadbandSlowWave` in
<basename>.SleepState.states.mat -> detectorinfo.detectionparms.SleepScoreMetrics.thratio, on the same
1 s `t_clus` grid. Definition (neurocode/brainStates/SleepScoreMaster/ClusterStates_GetMetrics.m,
lines 153-218, the IRASA branch these sessions use -- LFPparams IRASA = 1, SWweights 'PSS'): on the
theta channel (THchanID), 2 s windows, the IRASA oscillatory residual of the 2-20 Hz spectrum (the
aperiodic 1/f part removed), max over 5-10 Hz = theta power above the 1/f background; transients
removed, smoothed 15 s, min-max normalised to 0..1 within the session. Window value = mean of the
t_clus samples inside the window.

SleepScoreMaster scores THETA vs non-THETA wake from this same metric.

    -> data/revision/log/fig1_theta_ratio_windows.csv   (the UMAP's windows: session, state, center_s)
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

REPO = os.path.abspath(os.path.join(_HERE, ".."))
LOG = os.path.join(REPO, "data", "revision", "log")
OUT = os.path.join(LOG, "fig1_theta_ratio_windows.csv")


def load_theta_ratio(basepath):
    """(t_clus, thratio) on buzcode's 1 s grid, sorted, NaNs dropped."""
    path = _find_file(basepath, ".SleepState.states.mat")
    if path is None:
        raise FileNotFoundError(f"no SleepState.states.mat in {basepath}")
    msm = _loadmat_any(path)["SleepState"]["detectorinfo"]["detectionparms"]["SleepScoreMetrics"]
    t = np.asarray(msm["t_clus"], float).ravel()
    th = np.asarray(msm["thratio"], float).ravel()
    if t.size != th.size:
        raise ValueError(f"t_clus/thratio length mismatch in {path}")
    ok = np.isfinite(t) & np.isfinite(th)
    o = np.argsort(t[ok])
    return t[ok][o], th[ok][o]


def theta_ratio_window(t, th, t0, t1):
    lo, hi = np.searchsorted(t, (t0, t1))
    return (float(np.mean(th[lo:hi])), int(hi - lo)) if hi > lo else (np.nan, 0)


def main():
    F = pd.read_csv(os.path.join(LOG, "fig1_state_space_features.csv"), usecols=["session", "state", "center_s"])
    rows = []
    for bp in FIG1_SESSIONS:
        a, d = session_id(bp)
        sess = f"{a}_{d}_CA1"
        W = F[F.session == sess]
        if W.empty:
            continue
        t, th = load_theta_ratio(bp)
        for r in W.itertuples(index=False):
            v, n = theta_ratio_window(t, th, r.center_s - WINDOW_S / 2, r.center_s + WINDOW_S / 2)
            rows.append(dict(session=sess, state=r.state, center_s=r.center_s, theta_ratio=v, theta_n=n))
        g = pd.DataFrame(rows[-len(W):])
        print(f"{sess}: {len(W)} windows, grid {np.median(np.diff(t)):.2f} s, "
              + ", ".join(f"{s} {g[g.state == s].theta_ratio.median():.2f}" for s in ("THETA", "nonTHETA", "REM", "NREM")),
              flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(assert_local_path(OUT), index=False)
    print(f"-> {OUT}: {len(D)} windows, {D.theta_ratio.notna().mean():.1%} with a value")
    print(D.groupby("state").theta_ratio.describe()[["count", "25%", "50%", "75%"]].round(3).to_string())


if __name__ == "__main__":
    main()
