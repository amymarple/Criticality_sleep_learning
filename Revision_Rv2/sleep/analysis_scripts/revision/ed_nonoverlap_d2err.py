"""
ed_nonoverlap_d2err.py -- the two input tables of FigS1_C_SNR_S2C_d2 uncertainty.ipynb
(ED1c, ED2c) rebuilt on disjoint windows, schema unchanged, so the unchanged notebook
reads them in place of the shipped ones.

ed_nonoverlap.py thins every shipped window-level table that carries a window time;
these two carry none, so it reads them unchanged. ED2c's d2_norm, d2_err_norm and tile
are per-file transforms computed over all windows, so they are recomputed on the
disjoint windows rather than filtered by dropping rows.
  ED1c  d2_err_30s_60s_18sessions.csv  d2 (2024 Euclidean, AR(20), 40 ms) and its delta-
        method d2_err per window; 30 s windows (2 s step, 18 Fig-1 sessions) and 60 s
        windows (5 s step, the 11 sessions that have them). Columns state, d2, d2_err,
        region, animal, day, window.
  ED2c  d2_err_by_swr_rate_sextile.csv  NREM windows of the 12 barrage-cohort sessions
        (data/ripple_barr/d2_uncertainty/*_CA1.pkl, from the 30 s _error_bar.mat by
        MUA_sleepstage.ipynb): d2 z-scored per file (ddof 0), d2_err / that file's d2 SD,
        and the per-file SWR-rate sextile. Columns file_id, state, rate_col, d2_norm,
        d2_err_norm, ripple_rate_hz, tile.

The producer functions are exec'd from the figure notebook itself
(Previous_Rv1/sleep/figure_scripts/FigS1_C_SNR_S2C_d2 uncertainty.ipynb; identical in
first_submission_figures/). Each table is built twice: once on all windows, once with the
disjoint rule injected at load, so the producer's own state filter, z-score and tiling
then run on what is left.

Reads W: and Previous_Rv1 read-only; writes only the two tables below.

    -> data/revision/log/ed_nonoverlap_tables/d2_err_30s_60s_18sessions.csv
    -> data/revision/log/ed_nonoverlap_tables/d2_err_by_swr_rate_sextile.csv

    ENV=C:/Users/Cornell/.conda/envs/criticality/python.exe
    PYTHONIOENCODING=utf-8 $ENV ed_nonoverlap_d2err.py
"""
from __future__ import annotations

import os
import sys
import glob
import json
import pickle as _pickle
import re
import warnings
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path, FIG1_SESSIONS  # noqa: E402
import fig1de_nonoverlap as _f1de  # noqa: E402   (select_disjoint: the driver's rule, for the cross-check)

REPO = Path(_HERE).parents[0]
RV1 = Path(r"D:\Documents\GitHub\Criticality_sleep_learning\Previous_Rv1\sleep")
NB = RV1 / "figure_scripts" / "FigS1_C_SNR_S2C_d2 uncertainty.ipynb"
SHIPPED = RV1 / "csv" / "FigS1_C_SNR_S2C_d2 uncertainty"
PKL_DIR = REPO / "data" / "ripple_barr" / "d2_uncertainty"
OUTDIR = REPO / "data" / "revision" / "log" / "ed_nonoverlap_tables"
ED1C = "d2_err_30s_60s_18sessions.csv"
ED2C = "d2_err_by_swr_rate_sextile.csv"
TOL = 1e-6          # s; the same guard as select_disjoint
ATOL = 1e-9         # reconstruction tolerance on floats


# ------------------------------------------------------------------ the rule
def keep_disjoint(t, L):
    """Boolean mask, in input order: greedy over time, keep a window if its centre is >= L after the
    last kept one (first window always kept)."""
    t = np.asarray(t, float)
    assert np.all(np.isfinite(t)), "non-finite window centre"
    keep = np.zeros(t.size, bool)
    last = -np.inf
    for j in np.argsort(t, kind="stable"):
        if t[j] >= last + L - TOL:
            keep[j], last = True, t[j]
    return keep


def crosscheck_rule(key, t, L, keep):
    """keep_disjoint == fig1de_nonoverlap.select_disjoint (WIN_S set to L for the call), and the kept
    windows really are >= L apart within each key."""
    W = pd.DataFrame({"_key": key, "_t": np.asarray(t, float), "_i": np.arange(len(t))})
    old = _f1de.WIN_S
    try:
        _f1de.WIN_S = float(L)
        ref = _f1de.select_disjoint(W, key="_key", time_col="_t")
    finally:
        _f1de.WIN_S = old
    assert set(ref["_i"]) == set(np.flatnonzero(keep)), f"rule disagrees with select_disjoint (L={L})"
    K = W[keep].sort_values(["_key", "_t"])
    gaps = K.groupby("_key")["_t"].diff().dropna()
    assert (gaps >= L - TOL).all(), "kept windows closer than L"


# ------------------------------------------------------------------ the producer, from the notebook
def notebook_namespace():
    """exec the notebook's own producer code (and its `paths` list) into one namespace."""
    nb = json.load(open(NB, encoding="utf-8"))
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    ns = {"__name__": "figS1c_producer", "os": os, "re": re, "glob": glob, "np": np, "pd": pd,
          "Path": Path, "pickle": _pickle}
    for a, b in (("def _load_mat_any", "def plot_box_by_window"),       # ED1c loader + helpers
                 ("paths = [", "# === Main pooling"),                   # ED1c session list
                 ("def _force_sextiles", "# --- 3. Run Analysis ---")): # ED2c loader
        hits = [s for s in code if a in s and b in s]
        assert len(hits) == 1, f"notebook block {a!r} not found exactly once"
        exec(hits[0][hits[0].index(a):hits[0].index(b)], ns)
    ns["_orig_load_mat_any"] = ns["_load_mat_any"]
    return ns


def read_shipped(name):
    return pd.read_csv(SHIPPED / name, float_precision="round_trip")


def compare(new, old, name, sort_by=None):
    """Same columns, rows, strings and ints exactly, floats within ATOL. Returns max |diff| per float col."""
    assert list(new.columns) == list(old.columns), f"{name}: columns {list(new.columns)} != {list(old.columns)}"
    assert len(new) == len(old), f"{name}: {len(new)} rows rebuilt vs {len(old)} shipped"
    a, b = new.reset_index(drop=True), old.reset_index(drop=True)
    if sort_by is not None:
        a, b = a.sort_values(sort_by).reset_index(drop=True), b.sort_values(sort_by).reset_index(drop=True)
    diffs = {}
    for c in a.columns:
        if pd.api.types.is_float_dtype(b[c]):
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            assert np.array_equal(np.isnan(x), np.isnan(y)), f"{name}.{c}: NaN pattern differs"
            d = float(np.nanmax(np.abs(x - y))) if np.isfinite(x).any() else 0.0
            assert d <= ATOL, f"{name}.{c}: max |diff| {d:.3g} > {ATOL}"
            diffs[c] = d
        else:
            assert (a[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).all(), f"{name}.{c}: values differ"
    return diffs


# ------------------------------------------------------------------ ED1c
MAT_CACHE: dict = {}
ED1C_LOG: list = []


def win_len(mat_path):
    base = os.path.basename(mat_path)
    if base.endswith("_D2_time_resolved_error_bar_60sec.mat"):
        return 60.0
    if base.endswith("_D2_time_resolved_error_bar.mat"):
        return 30.0
    raise ValueError(f"unexpected d2 file {base}")


def thin_mat(mat, L, tag):
    """The .mat dict with every results[] entry cut to the disjoint windows (all states pooled)."""
    R = mat["results"]
    assert isinstance(R, list), "results is not a list of state structs"
    parts = [(i, np.atleast_1d(np.asarray(r.get("center", []), float))) for i, r in enumerate(R)]
    for (i, c), r in zip(parts, R):
        for f in ("d2", "d2_err"):
            v = np.atleast_1d(np.asarray(r.get(f, []), float))
            assert v.size == c.size, f"{tag}: {f} and center differ in length"
            assert np.all(np.isfinite(v)), f"{tag}: non-finite {f} (the producer would drop it; rule ambiguous)"
    t = np.concatenate([c for _, c in parts]) if parts else np.array([])
    owner = np.concatenate([np.full(c.size, i) for i, c in parts]) if parts else np.array([], int)
    keep = keep_disjoint(t, L)
    crosscheck_rule(np.zeros(t.size, int), t, L, keep)
    newR = []
    for i, r in enumerate(R):
        k = keep[owner == i]
        nr = dict(r)
        for f in ("center", "d2", "d2_err"):
            nr[f] = np.atleast_1d(np.asarray(r.get(f, []), float))[k]
        newR.append(nr)
    out = dict(mat)
    out["results"] = newR
    st = {str(r["state"]).upper(): (int((owner == i).sum()), int(keep[owner == i].sum())) for i, r in enumerate(R)}
    ED1C_LOG.append(dict(file=tag, L=L, n_all=int(t.size), n_disjoint=int(keep.sum()), **{
        f"{s}_all": v[0] for s, v in st.items()}, **{f"{s}_disjoint": v[1] for s, v in st.items()}))
    return out


def build_ed1c(ns, thin):
    def _load(mat_path):
        if mat_path not in MAT_CACHE:
            MAT_CACHE[mat_path] = ns["_orig_load_mat_any"](mat_path)
        m = MAT_CACHE[mat_path]
        if not thin:
            return m
        p = Path(mat_path)
        return thin_mat(m, win_len(mat_path), f"{p.parents[2].name}_{p.parents[1].name}_{int(win_len(mat_path))}s")
    ns["_load_mat_any"] = _load
    pooled = []
    for basepath in ns["paths"]:                         # the notebook's loop, failing loudly
        df = ns["load_d2_and_err_30s_60s_from_session"](basepath, "CA1")
        assert not df.empty, f"no d2 for {basepath}"
        pooled.append(df)
    return pd.concat(pooled, ignore_index=True)


# ------------------------------------------------------------------ ED2c
ED2C_LOG: list = []


class ThinningPickle:
    """Stands in for `pickle` inside load_pool_rate_binned: loads the file, keeps the disjoint windows
    (all states, time order, original row order and index), then hands the frame to the producer."""

    def __init__(self, L):
        self.L = L

    def load(self, f):
        df = _pickle.load(f)
        assert isinstance(df, pd.DataFrame) and "center_s" in df.columns
        assert not df[["center_s", "d2", "d2_err", "ripple_rate_hz"]].isna().any().any(), "NaN in source pkl"
        keep = keep_disjoint(df["center_s"], self.L)
        crosscheck_rule(np.zeros(len(df), int), df["center_s"], self.L, keep)
        d = df[keep]
        ED2C_LOG.append(dict(file_id=Path(f.name).stem, n_all=len(df), n_disjoint=len(d),
                             NREM_all=int((df.state == "NREM").sum()), NREM_disjoint=int((d.state == "NREM").sum()),
                             WAKE_disjoint=int((d.state == "WAKE").sum()), REM_disjoint=int((d.state == "REM").sum())))
        return d


def build_ed2c(ns, thin):
    ns["pickle"] = ThinningPickle(30.0) if thin else _pickle     # the pkls hold 30 s windows (_error_bar.mat)
    try:
        out = ns["load_pool_rate_binned"](PKL_DIR, rate_col="ripple_rate_hz", d2_col="d2",
                                          d2_err_col="d2_err", state="NREM", n_bins=6)
    finally:
        ns["pickle"] = _pickle
    files = sorted(p.stem for p in PKL_DIR.glob("*.pkl"))
    assert sorted(out.file_id.unique()) == files, "a file was skipped by the producer"
    return out


def tiling_report(d):
    """Per file: whether plain qcut(duplicates='drop') gave 6 bins, or _force_sextiles fell back to ranks."""
    rows = []
    for fid, g in d.groupby("file_id", sort=False):
        q = pd.qcut(g.ripple_rate_hz, 6, labels=False, duplicates="drop")
        rows.append(dict(file_id=fid, n=len(g), qcut_bins=int(q.nunique()),
                         plain_qcut_matches=bool((q + 1 == g.tile.astype(int)).all()),
                         tiles=g.tile.value_counts().sort_index().tolist()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    pd.set_option("display.width", 220)
    ns = notebook_namespace()
    assert {(Path(p).parent.name, Path(p).name) for p in ns["paths"]} == \
        {(Path(p).parent.name, Path(p).name) for p in FIG1_SESSIONS}, "notebook paths != FIG1_SESSIONS"

    # ---------- ED1c
    print("=== ED1c", ED1C)
    A1 = build_ed1c(ns, thin=False)
    d1 = compare(A1, read_shipped(ED1C), ED1C)
    print(f"  ALL windows reproduce the shipped table: {len(A1)} rows, max |diff| {d1}")
    D1 = build_ed1c(ns, thin=True)
    assert list(D1.columns) == list(A1.columns)
    L1 = pd.DataFrame(ED1C_LOG)
    print(L1.to_string(index=False))
    n_all = A1.groupby(["window"]).size()
    n_dis = D1.groupby(["window"]).size()
    print("  rows by window, all -> disjoint:", {w: f"{n_all[w]} -> {n_dis[w]}" for w in n_all.index})
    print("  rows by window x state (disjoint):\n", D1.groupby(["window", "state"]).size().to_string())

    # ---------- ED2c
    print("\n=== ED2c", ED2C)
    A2 = build_ed2c(ns, thin=False)
    S2 = read_shipped(ED2C)
    d2 = compare(A2, S2, ED2C)
    print(f"  ALL windows reproduce the shipped table: {len(A2)} rows, max |diff| {d2}")
    T = tiling_report(A2)
    print("  tiling on ALL windows (shipped):\n", T.to_string(index=False))
    D2 = build_ed2c(ns, thin=True)
    assert list(D2.columns) == list(A2.columns)
    print(pd.DataFrame(ED2C_LOG).to_string(index=False))
    print("  tiling on DISJOINT windows:\n", tiling_report(D2).to_string(index=False))
    zchk = D2.groupby("file_id").d2_norm.agg(["mean", lambda s: s.std(ddof=0)])
    assert np.allclose(zchk.iloc[:, 0], 0, atol=1e-9) and np.allclose(zchk.iloc[:, 1], 1, atol=1e-9), "re-z failed"

    # ---------- write, then read back the way the notebook does
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for name, D, A in ((ED1C, D1, A1), (ED2C, D2, A2)):
        path = str(OUTDIR / name)
        D.to_csv(assert_local_path(path), index=False)
        back = pd.read_csv(path, float_precision="round_trip")
        compare(back, D.reset_index(drop=True), name + " (read back)")
        assert list(back.columns) == list(read_shipped(name).columns)
        assert all(back[c].dtype == read_shipped(name)[c].dtype for c in back.columns), f"{name}: dtype changed"
        print(f"wrote {path}  ({len(A)} -> {len(D)} rows)")


if __name__ == "__main__":
    main()
