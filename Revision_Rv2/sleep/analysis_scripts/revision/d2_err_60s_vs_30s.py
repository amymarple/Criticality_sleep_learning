"""
d2_err_60s_vs_30s.py -- d2 (2024 Euclidean, p = 20, 40 ms) and its delta-method relative uncertainty at 60 s and
at 30 s windows, by state, for comparison with the method papers at their window length (Sooter et al. used 60 s).

SOURCE. The MATLAB producer's own outputs on the share, read-only:
    <day>_CA1_D2_time_resolved_error_bar.mat         30 s windows, 2 s step
    <day>_CA1_D2_time_resolved_error_bar_60sec.mat   60 s windows
Both carry d2 and d2_err (getDbetas_error_bar.m, delta method on the AR fit). States are the .mat's
NREM / REM / WAKE (WAKE = theta + non-theta; Sooter's "awake"). Windows whose centres are >= 30 s (30 s windows)
or >= 60 s (60 s windows) apart within each state are used, so that contents are disjoint; per session the median
of d2 and of d2_err / d2 per state, then the median [IQR] across sessions. 18 Fig-1 sessions.
`read` and `thin` are also used by r24_state_centre.py and d2_wake_env_audit.py.

Reads the share read-only; writes only to D:.
    -> data/revision/log/d2_err_60s_vs_30s.csv
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.io import loadmat

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path, FIG1_SESSIONS, session_id   # noqa: E402

LOGDIR = os.path.abspath(os.path.join(_HERE, "..", "data", "revision", "log"))
OUT = os.path.join(LOGDIR, "d2_err_60s_vs_30s.csv")


def read(bp, tag):
    day = os.path.basename(bp.rstrip("\\/"))
    f = os.path.join(bp, "complex_system", f"{day}_CA1_D2_time_resolved{tag}.mat")
    if not os.path.exists(f):
        return None
    m = loadmat(f, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    out = []
    for r in np.atleast_1d(m["results"]):
        st = str(r["state"]).upper()
        c = np.atleast_1d(r.get("center", np.nan)).astype(float)
        d = np.atleast_1d(r.get("d2", np.nan)).astype(float)
        e = np.atleast_1d(r.get("d2_err", np.full_like(d, np.nan))).astype(float)
        if d.size and c.size == d.size:
            out.append(pd.DataFrame({"state": st, "center": c, "d2": d, "err": e if e.size == d.size else np.nan}))
    return pd.concat(out, ignore_index=True) if out else None


def thin(T, spacing):
    """keep windows whose centres are >= spacing apart within each state (disjoint contents)"""
    keep = []
    for st, g in T.sort_values("center").groupby("state"):
        last = -np.inf
        for i, c in zip(g.index, g.center):
            if c - last >= spacing:
                keep.append(i); last = c
    return T.loc[keep]


def main():
    rows = []
    for bp in FIG1_SESSIONS:
        animal, day = session_id(bp); sid = f"{animal}_{day}"
        for win, tag in ((30, "_error_bar"), (60, "_error_bar_60sec")):
            T = read(bp, tag)
            if T is None:
                print(f"  [missing] {sid} {win} s"); continue
            T = thin(T.dropna(subset=["d2"]), win)
            T["rel"] = T.err / T.d2
            for st, g in T.groupby("state"):
                rows.append(dict(session=sid, window_s=win, state=st, n=len(g), d2=g.d2.median(), err=g.err.median(), rel=g.rel.median()))
        print(f"  {sid} done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(OUT), index=False)
    print("\nper state: median [IQR] across sessions of the per-session medians (non-overlapping windows)")
    for (win, st), g in R.groupby(["window_s", "state"]):
        q = lambda c: f"{g[c].median():.4f} [{g[c].quantile(.25):.4f}, {g[c].quantile(.75):.4f}]"
        print(f"  {win:2d} s  {st:5s} n={len(g):2d}  d2 {q('d2')}  d2_err {q('err')}  err/d2 {100*g.rel.median():.1f} % [{100*g.rel.quantile(.25):.1f}, {100*g.rel.quantile(.75):.1f}]")


if __name__ == "__main__":
    main()
