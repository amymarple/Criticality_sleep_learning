"""
d2_wake_env_audit.py -- wake theta vs non-theta d2 (2024 Euclidean, p = 20, 40 ms) under the two wake labellings
and by epoch environment.

Fig 1D (`Fig1_D_histogram_S1_E.ipynb`, `plot_density_split`) compares THETA vs nonTHETA only among wake windows
whose centre lies in a NON-sleep epoch (`environment != 'sleep'`, from `loading.load_epoch`), on every 30 s window
of the MATLAB producer's `_error_bar.mat` (2 s step); the Fig-1 tiles (fig1_session_medians.csv) use all wake and
the whole-window labelling.

DESIGN. Per Fig-1 session: the `_error_bar.mat` WAKE windows, split exactly as the notebook does
(`split_wake_into_theta_nontheta`: THETA if the window centre is in SleepState THETA, else nonTHETA;
`annotate_with_epochs`: environment of the epoch holding the centre). Per-session median d2 for THETA / nonTHETA
in (a) all wake, (b) env != sleep (the Fig 1D panel), (c) env == sleep (wake in the sleep box); every window and
non-overlapping windows (every 15th of the 2 s grid). Paired Wilcoxon across sessions having both.
`in_intervals` is also used by r24_state_centre.py.

Reads the share read-only; writes only to D:.
    -> data/revision/log/d2_wake_env_audit.csv
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path, FIG1_SESSIONS, session_id   # noqa: E402
from d2_err_60s_vs_30s import read as read_mat                              # noqa: E402
from fig1_metrics import build_state_intervals                              # noqa: E402

LOGDIR = os.path.abspath(os.path.join(_HERE, "..", "data", "revision", "log"))
OUT = os.path.join(LOGDIR, "d2_wake_env_audit.csv")


def in_intervals(t, iv):
    iv = np.atleast_2d(np.asarray(iv, float))
    if iv.size == 0:
        return np.zeros(t.size, bool)
    iv = iv[np.argsort(iv[:, 0])]
    j = np.searchsorted(iv[:, 0], t, "right") - 1
    ok = j >= 0
    out = np.zeros(t.size, bool)
    out[ok] = t[ok] < iv[j[ok], 1]
    return out


def inside(c, iv, half=15.0):
    """window [c - half, c + half] lies entirely inside one interval of iv"""
    iv = np.atleast_2d(np.asarray(iv, float))
    if iv.size == 0:
        return np.zeros(c.size, bool)
    iv = iv[np.argsort(iv[:, 0])]
    j = np.searchsorted(iv[:, 0], c - half + 1e-9, "right") - 1
    ok = j >= 0
    out = np.zeros(c.size, bool)
    out[ok] = (c[ok] + half) <= iv[j[ok], 1] + 1e-9
    return out


def environment(t, ep):
    ep = ep.sort_values("startTime")
    s, e = ep.startTime.to_numpy(float), ep.stopTime.to_numpy(float)
    env = ep.get("environment", pd.Series([""] * len(ep))).astype(str).str.lower().to_numpy()
    j = np.searchsorted(s, t, "right") - 1
    out = np.full(t.size, None, dtype=object)
    ok = (j >= 0) & (t < e[np.clip(j, 0, None)])
    out[ok] = env[j[ok]]
    return out


def main():
    from neuro_py.io import loading
    rows = []
    for bp in FIG1_SESSIONS:
        a, d = session_id(bp); sid = f"{a}_{d}"
        M = read_mat(bp, "_error_bar")
        if M is None:
            print(f"  [missing .mat] {sid}"); continue
        W = M[M.state == "WAKE"].dropna(subset=["d2"]).sort_values("center").copy()
        th = loading.load_SleepState_states(bp).get("THETA", np.zeros((0, 2)))
        W["sub"] = np.where(in_intervals(W.center.to_numpy(float), th), "THETA", "nonTHETA")
        W["env"] = environment(W.center.to_numpy(float), loading.load_epoch(bp))
        W["nonoverlap"] = False
        W.loc[W.index[::15], "nonoverlap"] = True
        # the Fig-1 tiles' labelling: the whole 30 s inside WAKE & THETA (or WAKE - THETA), both
        # guarded by 5 s (fig1_metrics.build_state_intervals); the rest straddle a transition
        iv = build_state_intervals(loading.load_SleepState_states(bp))
        c = W.center.to_numpy(float)
        W["tile"] = np.where(inside(c, iv["THETA"]), "THETA", np.where(inside(c, iv["nonTHETA"]), "nonTHETA", "straddle"))
        for lab, col in (("centre label", "sub"), ("whole-window label", "tile")):
            for sub in ("THETA", "nonTHETA", "straddle"):
                g = W[W[col] == sub]
                if len(g):
                    rows.append(dict(session=sid, subset=lab, windows="every window", state=sub, n=len(g), d2=g.d2.median()))
        for sel, lab in ((W.env.notna(), "all wake"), (W.env.notna() & (W.env != "sleep"), "env != sleep"),
                         (W.env == "sleep", "env == sleep")):
            for win, m in (("every window", sel), ("non-overlapping", sel & W.nonoverlap)):
                for sub in ("THETA", "nonTHETA"):
                    g = W[m & (W["sub"] == sub)]
                    rows.append(dict(session=sid, subset=lab, windows=win, state=sub, n=len(g),
                                     d2=g.d2.median() if len(g) else np.nan))
        envs = W.env.value_counts(dropna=False).to_dict()
        print(f"  {sid:12s} wake windows {len(W):5d}  env: {envs}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(OUT), index=False)
    print("\nper-session median d2 (2024 Euclidean), THETA vs nonTHETA, paired Wilcoxon (sessions with >= 5 windows of each)")
    for (lab, win), g in R.groupby(["subset", "windows"], sort=False):
        P = g[(g.n >= 5) & g.state.isin(["THETA", "nonTHETA"])].pivot(index="session", columns="state", values="d2").dropna()
        if len(P) < 3:
            print(f"  {lab:13s} {win:16s} n = {len(P)}"); continue
        dlt = P.THETA - P.nonTHETA
        print(f"  {lab:13s} {win:16s} n = {len(P):2d}  THETA {P.THETA.median():.4f}  nonTHETA {P.nonTHETA.median():.4f}  "
              f"THETA < nonTHETA in {(dlt < 0).sum()}/{len(P)}  p = {stats.wilcoxon(dlt).pvalue:.4f}")
    for lab in ("centre label", "whole-window label"):
        g = R[(R.subset == lab)]
        tot = g.groupby("state").n.sum()
        med = g.groupby("state").d2.median()
        print(f"  {lab:19s} windows: " + ", ".join(f"{s} {int(tot.get(s, 0))} (median of session medians {med.get(s, np.nan):.4f})"
                                                   for s in ("THETA", "nonTHETA", "straddle")))


if __name__ == "__main__":
    main()
