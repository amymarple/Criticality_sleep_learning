"""
ed_nonoverlap_ed1b.py -- ED1b (variance explained by the AR(20) fit vs window length) on disjoint
windows, rebuilt from the MATLAB producer's own output.

The table explained_variance_by_window.csv has no window times. The MATLAB producer records
them: coarse_grain_ddc.m (matlab/; from the lab's R_Project repository) slides windows of
`window_size` s by `step_size` s over NREM, fits getDbetas (AR 20, 40 ms bins) and saves,
per session x window length, `centers`, `explain_variance`, `D2_values` into
20250324timebin_grainedmouse.mat. The steps there are 0.25 s (0.5 s and 1 s windows),
0.5 s (5 s), 1 s (10 s), 3 s (30 s) and 5 s (60 s), so every window overlaps its neighbours.

Method:
  1. The FigS1_B notebook's original loader (fields grouped by the `win(\d+)` key in field order; one
     list entry per session) rebuilds the shipped CSV from the .mat on all windows.
  2. Within each session x window length, windows in time order, one kept only if its centre is
     >= window_size after the last kept one (the windows are NREM only, so this is the disjoint rule of
     fig1de_nonoverlap at L = window length).
  3. The thinned table is written under the shipped name to ed_nonoverlap_tables/, where
     ed_nonoverlap.py reads it in place of the shipped CSV.

    -> data/revision/log/ed_nonoverlap_tables/explained_variance_by_window.csv
    -> data/revision/log/ed1b_thinning.csv   (rows per session x window length, all -> disjoint)
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
import pandas as pd
import scipy.io as sio

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
MAT = r"D:\Documents\GitHub\R_Project\PPP\codebook\Criticality_analysis\ddc\data\coarse_grain_bin_explore\20250324timebin_grainedmouse.mat"
SHIPPED = (r"D:\Documents\GitHub\Criticality_sleep_learning\Previous_Rv1\sleep\csv"
           r"\FigS1_B_Explain variance_D_statetransition\explained_variance_by_window.csv")
OVR = os.path.join(REPO, "data", "revision", "log", "ed_nonoverlap_tables", "explained_variance_by_window.csv")
LOG = os.path.join(REPO, "data", "revision", "log", "ed1b_thinning.csv")


def disjoint(c, L):
    order = np.argsort(c, kind="stable")
    keep = np.zeros(c.size, bool)
    last = -np.inf
    for j in order:
        if c[j] >= last + L - 1e-9:
            keep[j], last = True, c[j]
    return keep


def build(thin):
    m = sio.loadmat(MAT, squeeze_me=True, struct_as_record=False)
    groups = {}                                   # the notebook's groupData: key -> [session structs]
    for f in [k for k in m if not k.startswith("__")]:
        mm = re.search(r"win(\d+)", f)
        if mm:
            groups.setdefault(f"win{int(mm.group(1))}", []).append((f, m[f]))
    rows, log = [], []
    for key, sess in groups.items():
        for i, (f, a) in enumerate(sess):
            ev = np.atleast_1d(a.explain_variance).astype(float).ravel()
            c = np.atleast_1d(a.centers).astype(float).ravel()
            assert ev.size == c.size, f"{f}: {ev.size} values vs {c.size} centres"
            keep = disjoint(c, float(a.window_size)) if thin else np.ones(c.size, bool)
            rows.append(pd.DataFrame({"window": key, "animal": i, "explain_variance": ev[keep]}))
            log.append(dict(field=f, window=key, window_size_s=float(a.window_size), session_index=i,
                            rows_all=c.size, rows_disjoint=int(keep.sum())))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(log)


def main():
    S = pd.read_csv(SHIPPED, float_precision="round_trip")
    A, _ = build(thin=False)
    assert list(A.columns) == list(S.columns) and len(A) == len(S), (A.shape, S.shape)
    assert (A.window.values == S.window.values).all() and (A.animal.values == S.animal.values).all()
    ok = np.isclose(A.explain_variance.values, S.explain_variance.values, atol=1e-12, equal_nan=True).all()
    assert ok, "explain_variance differs from the shipped table"
    print(f"rebuild on all windows = shipped table: {len(A)} rows, identical")
    D, L = build(thin=True)
    os.makedirs(os.path.dirname(OVR), exist_ok=True)
    D.to_csv(assert_local_path(OVR), index=False)
    L.to_csv(assert_local_path(LOG), index=False)
    print(L.groupby(["window", "window_size_s"])[["rows_all", "rows_disjoint"]].sum().to_string())
    print(f"wrote {OVR} ({len(D)} rows)")


if __name__ == "__main__":
    main()
