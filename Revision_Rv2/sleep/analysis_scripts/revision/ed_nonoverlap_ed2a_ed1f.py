"""
ed_nonoverlap_ed2a_ed1f.py -- the input tables of ED2a (novel vs familiar d2, relative to
pre-task sleep) and ED1f (branching-ratio distribution by state) rebuilt on disjoint
windows, for ed_nonoverlap.py to read in place of the shipped ones.

ed_nonoverlap.py thins every window-level table a Previous_Rv1 figure notebook reads,
but the two tables behind ED2a (d2_rel_novel_familiar.csv) and ED1f
(branching_ratio_by_state.csv) carry no window time, so the driver cannot thin them.
They are rebuilt here from their producers (MUA_sleepstage.ipynb for ED2a,
FigS1_F_branchingratio_histogram.ipynb for ED1f) with the disjoint rule applied before
the per-session reference (ED2a: the pre-task-sleep mean) and the weights (ED1f).

The producers' own cells are executed from the notebook source, with asserted
one-line patches that insert the thinning; each table is rebuilt on all windows,
then on disjoint windows. A third ED1f variant keeps THETA/nonTHETA windows that fall
in a sleep-labelled epoch (the producer otherwise drops them), since the adjacent panels
use every wake window; it is written to its own output folder.

Disjoint rule (fixed by the caller): within one session, the windows of every state in
time order; a window is kept only if its centre is >= L after the last kept window
(L = window length; 30 s for ED2a, 10 s for ED1f).

    -> data/revision/log/ed_nonoverlap_tables/d2_rel_novel_familiar.csv
    -> data/revision/log/ed_nonoverlap_tables/branching_ratio_by_state.csv
    -> data/revision/log/ed_nonoverlap_tables_allwake/branching_ratio_by_state.csv
    -> --rows-csv PATH (optional): rows kept per session and table, all -> disjoint (-> all wake for ED1f)

    ENV=C:/Users/Cornell/.conda/envs/criticality/python.exe
    $ENV ed_nonoverlap_ed2a_ed1f.py [--rows-csv PATH] [--ed1f-only]
"""
from __future__ import annotations

import os
import sys
import argparse
import contextlib
import glob
import io
import json
import pickle
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig1de_nonoverlap import select_disjoint  # noqa: E402
from run_fig1_metrics import assert_local_path  # noqa: E402
from neuro_py.io import loading  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
RV1 = r"D:\Documents\GitHub\Criticality_sleep_learning\Previous_Rv1\sleep"
OUT_DIR = os.path.join(REPO, "data", "revision", "log", "ed_nonoverlap_tables")
# ED1f with every wake window (Extended Data Fig. 2e, since the adjacent panels use every wake
# window): the producer's sleep-environment drop of THETA / nonTHETA removed; its own folder,
# read by run_rv1_capture(override_dir)
OUT_DIR_ALLWAKE = os.path.join(REPO, "data", "revision", "log", "ed_nonoverlap_tables_allwake")
PPP = r"W:\data\PPP"
TOL = 1e-9

# ---- ED2a ----
NB_MUA = os.path.join(REPO, "session_tables", "MUA_sleepstage.ipynb")
NF_DIR = os.path.join(REPO, "data", "novel_familiar")
NF_FOLDERS = ["control", "novel"]                          # FigS2_A_novel_familar cell 1, in this order
ED2A_NAME = "d2_rel_novel_familiar.csv"
ED2A_SHIPPED = os.path.join(RV1, "csv", "FigS2_A_novel_familar", ED2A_NAME)
# the stored pickles of these two were made from day 12's d2 (and the session's own epochs)
D2_FROM_DAY = {"PPP7_day14": "day12", "PPP8_day14": "day12"}
DROP_DISJOINT = set(D2_FROM_DAY)
L_D2 = 30.0
CAP_N = 1200                                               # cell 15: first_n_per_epoch(d, n=1200)

# ---- ED1f ----
NB_BR = os.path.join(REPO, "first_submission_figures", "FigS1_F_branchingratio_histogram.ipynb")
BR_DIR = os.path.join(REPO, "data", "cross_state", "branching_ratio_1")
ED1F_NAME = "branching_ratio_by_state.csv"
ED1F_SHIPPED = os.path.join(RV1, "csv", "FigS1_F_branchingratio_histogram", ED1F_NAME)
L_BR = 10.0


# ----------------------------------------------------------------------------- helpers
def uncomment(src):
    """A commented-out notebook cell back to code: drop one leading '# ' (or '#') per line."""
    return "\n".join(ln[2:] if ln.startswith("# ") else ln[1:] if ln.startswith("#") else ln
                     for ln in src.split("\n"))


def between(src, start, stop):
    assert src.count(start) == 1 and src.count(stop) == 1, f"markers {start!r} / {stop!r} not unique"
    return src[src.index(start): src.index(stop)]


def patched(src, old, new, n=1):
    assert src.count(old) == n, f"expected {n} x {old!r}, found {src.count(old)}"
    return src.replace(old, new)


def disjoint_mask(t, L):
    """Greedy over centres in time order (all states): keep a window if its centre is >= L after the
    last kept one. `t` must be strictly increasing (no two windows of a session share a centre)."""
    t = np.asarray(t, float)
    assert np.all(np.diff(t) > 0), "window centres not strictly increasing within the session"
    keep = np.zeros(t.size, bool)
    last = -np.inf
    for j, x in enumerate(t):
        if x >= last + L - 1e-6:
            keep[j], last = True, x
    return keep


def quiet(fn, *a, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


def assert_same_table(new, shipped, labels, values, name):
    """Row-by-row equality with the shipped CSV: columns, row count, labels exact, values within TOL."""
    assert list(new.columns) == list(shipped.columns), f"{name}: columns {list(new.columns)} != {list(shipped.columns)}"
    assert len(new) == len(shipped), f"{name}: {len(new)} rows rebuilt vs {len(shipped)} shipped"
    for c in labels:
        bad = int((new[c].astype(str).to_numpy() != shipped[c].astype(str).to_numpy()).sum())
        assert bad == 0, f"{name}: column {c} differs in {bad} rows"
    out = {}
    for c in values:
        a, b = new[c].to_numpy(float), shipped[c].to_numpy(float)
        assert np.array_equal(np.isnan(a), np.isnan(b)), f"{name}: NaN pattern of {c} differs"
        d = np.abs(a - b)[~np.isnan(a)]
        out[c] = float(d.max()) if d.size else 0.0
        assert out[c] <= TOL, f"{name}: max |diff| of {c} = {out[c]:.3g} > {TOL}"
    return out


def write_csv(df, path):
    assert_local_path(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    assert_local_path(path)
    df.to_csv(path, index=False)                            # same row order/index as the shipped CSVs


# ----------------------------------------------------------------------------- ED2a
def ed2a_producer():
    cells = json.load(open(NB_MUA, encoding="utf-8"))["cells"]
    c8 = uncomment("".join(cells[8]["source"]))
    c15 = uncomment("".join(cells[15]["source"]))
    ns = {}
    exec(c8[: c8.index("# Test the complete loading function")], ns)            # load_d2_timeline_complete
    exec(c15[: c15.index("# -------- 1) annotate ca1_complete_df")], ns)          # label_epoch_from_name, annotate_epochs
    exec(between(c15, "def first_n_per_epoch", "sleep_groups = "), ns)
    block = between(c15, "sleep_groups = ", "# ---------- plot")                  # NREM, cap, m_pre, d2_rel
    block = patched(block, "d = first_n_per_epoch(d, n=1200)", "d = _CAP(d)")
    return ns, block


def run_cell15(ns, block, ca_annot, cap):
    loc = dict(ns, ca_annot=ca_annot, _CAP=cap)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(block, loc)
    return loc["d"], loc["d2_rel"], loc["m_pre"]


def stack(rel, stem, sub):
    """FigS2_A_novel_familar cell 1 (original loader) for one pickle."""
    use = rel[rel["epoch_group"].isin(["midsleep", "postsleep"])].copy()
    if use.empty:
        return None
    use["file"] = stem
    use["source"] = sub
    use["phase"] = np.where(use["epoch_group"].eq("midsleep"), "mid", "post")
    return use[["file", "source", "phase", "d2_rel"]]


def ed2a():
    ns, block = ed2a_producer()
    first_n = lambda d: ns["first_n_per_epoch"](d, n=CAP_N)  # noqa: E731
    rows_all, rows_dis, log = [], [], []
    for sub in NF_FOLDERS:
        for pkl in sorted(glob.glob(os.path.join(NF_DIR, sub, "*.pkl"))):
            stem = os.path.basename(pkl)[:-4]
            animal, day = stem.split("_")
            ca1, meta = quiet(ns["load_d2_timeline_complete"],
                              os.path.join(PPP, animal, D2_FROM_DAY.get(stem, day)), "CA1")
            assert os.path.basename(meta["path"]).endswith("_CA1_D2_time_resolved_error_bar.mat"), meta["path"]
            epoch_df = loading.load_epoch(os.path.join(PPP, animal, day))

            # rebuild on all windows
            d_all, rel_all, m_all = run_cell15(ns, block, ns["annotate_epochs"](ca1, epoch_df), first_n)
            ref = pd.read_pickle(pkl)
            assert len(ref) == len(rel_all), f"{stem}: {len(rel_all)} rows vs {len(ref)} in the stored pickle"
            for c in ("center_s", "d2", "d2_rel"):
                dmax = float(np.max(np.abs(ref[c].to_numpy(float) - rel_all[c].to_numpy(float)))) if len(ref) else 0.0
                assert dmax <= TOL, f"{stem}: {c} differs from the stored pickle by {dmax:.3g}"
            assert (ref["epoch_group"].astype(str).to_numpy() == rel_all["epoch_group"].astype(str).to_numpy()).all()
            s = stack(rel_all, stem, sub)
            if s is not None:
                rows_all.append(s)

            # disjoint: thin the whole timeline (every state), then cell 15 inside the producer's span
            t = ca1["center_s"].to_numpy(float)
            keep = disjoint_mask(t, L_D2)
            ref_keep = select_disjoint(ca1.assign(_k=stem), key="_k", time_col="center_s")["center_s"].to_numpy(float)
            assert np.array_equal(np.sort(ref_keep), t[keep]), "disjoint_mask disagrees with select_disjoint"
            annot_dis = ns["annotate_epochs"](ca1[keep], epoch_df)
            span = set(d_all["center_s"].to_numpy(float))
            in_span = lambda d: d[d["center_s"].isin(span)].sort_values("center_s")  # noqa: E731
            d_dis, rel_dis, m_dis = run_cell15(ns, block, annot_dis, in_span)
            assert set(rel_dis["center_s"]) <= set(rel_all["center_s"])
            _, rel_lit, m_lit = run_cell15(ns, block, annot_dis, first_n)  # literal cap: rows table only
            dropped = stem in DROP_DISJOINT
            if not dropped:
                s = stack(rel_dis, stem, sub)
                if s is not None:
                    rows_dis.append(s)

            n_pre = lambda d: int((d["epoch_group"] == "presleep").sum())  # noqa: E731
            for ph, g in (("mid", "midsleep"), ("post", "postsleep")):
                a = rel_all[rel_all["epoch_group"] == g]["d2_rel"]
                b = rel_dis[rel_dis["epoch_group"] == g]["d2_rel"]
                c = rel_lit[rel_lit["epoch_group"] == g]["d2_rel"]
                log.append(dict(table=ED2A_NAME, session=stem, source=sub, group=ph,
                                rows_all=len(a), rows_disjoint=0 if dropped else len(b),
                                dropped_duplicate=dropped,
                                presleep_ref_windows_all=n_pre(d_all), presleep_ref_windows_disjoint=n_pre(d_dis),
                                m_pre_all=m_all, m_pre_disjoint=m_dis,
                                mean_d2_rel_all=a.mean() if len(a) else np.nan,
                                mean_d2_rel_disjoint=b.mean() if len(b) else np.nan,
                                rows_disjoint_literal_cap=len(c),
                                mean_d2_rel_disjoint_literal_cap=c.mean() if len(c) else np.nan,
                                d2_mat=os.path.basename(meta["path"]),
                                d2_session=f"{animal}_{D2_FROM_DAY.get(stem, day)}"))
            print(f"  ED2a {sub:7s} {stem:11s} all {len(rel_all):5d} -> disjoint "
                  f"{'dropped (duplicate)' if dropped else len(rel_dis)}   m_pre {m_all:.6f} -> {m_dis:.6f}",
                  flush=True)
    return pd.concat(rows_all, ignore_index=True), pd.concat(rows_dis, ignore_index=True), pd.DataFrame(log)


# ----------------------------------------------------------------------------- ED1f
def ed1f_producer():
    src = "".join(json.load(open(NB_BR, encoding="utf-8"))["cells"][0]["source"])
    code = between(src, "OUTDIR = ", "# ---------- BUILD BINS")
    code = patched(code, 'pkl_paths = sorted(glob.glob(os.path.join(OUTDIR, "branching_ratio_allstates_*.pkl")))',
                   "pkl_paths = _PATHS")
    code = patched(code, "payload = pickle.load(f)", "payload = _PREP(pickle.load(f))")
    ns = dict(os=os, glob=glob, pickle=pickle, np=np, pd=pd, Path=Path, loading=loading)
    exec(between(src, "OUTDIR = ", "SAVE_PREFIX"), ns)
    assert os.path.normcase(os.path.abspath(ns["OUTDIR"])) == os.path.normcase(BR_DIR), ns["OUTDIR"]
    return code, ns


def thin_payload(payload, L):
    """The session's windows of every state, thinned in time order; each state keeps its kept windows
    (centres, branching ratios, Goodfit) in their own order. meta is untouched."""
    states = [k for k in payload if k != "meta"]
    cen = [np.asarray(payload[st]["centers"], float) for st in states]
    for st, c in zip(states, cen):
        b = np.asarray(payload[st]["branching_ratios"], float)
        # aligned only if centres and branching ratios are both finite (checked below)
        assert c.size == b.size and np.isfinite(c).all() and np.isfinite(b).all(), st
    allc = np.concatenate(cen)
    lab = np.concatenate([np.full(c.size, i) for i, c in enumerate(cen)])
    order = np.argsort(allc, kind="stable")
    keep = np.zeros(allc.size, bool)
    keep[order[disjoint_mask(allc[order], L)]] = True
    out = {"meta": payload["meta"]}
    for i, st in enumerate(states):
        idx = np.flatnonzero(keep[lab == i])
        n = cen[i].size
        out[st] = {k: ([v[j] for j in idx] if isinstance(v, (list, tuple, np.ndarray)) and len(v) == n else v)
                   for k, v in payload[st].items()}
    return out


def ed1f():
    code, ns = ed1f_producer()
    paths = sorted(glob.glob(os.path.join(BR_DIR, "branching_ratio_allstates_*.pkl")))   # the producer's glob
    assert paths and all("_win10s_step2s_guard5s_" in os.path.basename(p) for p in paths), paths
    states = ns["STATES"]
    code_allwake = patched(code, "mask &= ~in_sleep", "pass")                  # keep wake windows in sleep epochs
    parts = {arm: {st: [] for st in states} for arm in ("all", "disjoint", "disjoint_allwake")}
    log = []
    for p in paths:
        sess = os.path.basename(p).split("guard5s_")[1][:-4]
        meta = pickle.load(open(p, "rb"))["meta"]
        assert float(meta["window_size"]) == L_BR and float(meta["step_size"]) == 2.0, meta
        res = {}
        for arm, prep, src in (("all", lambda x: x, code), ("disjoint", lambda x: thin_payload(x, L_BR), code),
                               ("disjoint_allwake", lambda x: thin_payload(x, L_BR), code_allwake)):
            loc = dict(ns, _PATHS=[p], _PREP=prep)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(src, loc)
            res[arm] = (loc["pooled_vals"], loc["pooled_wts"])
            for st in states:
                parts[arm][st].append(pd.DataFrame({"state": st, "br": loc["pooled_vals"][st],
                                                    "weight": loc["pooled_wts"][st]}))
        for st in states:
            log.append(dict(table=ED1F_NAME, session=sess, source="", group=st,
                            rows_all=res["all"][0][st].size, rows_disjoint=res["disjoint"][0][st].size,
                            rows_disjoint_allwake=res["disjoint_allwake"][0][st].size, dropped_duplicate=False,
                            weight_values_disjoint=",".join(map(str, np.unique(res["disjoint"][1][st])))))
        print(f"  ED1f {sess:15s} " + "  ".join(f"{st} {res['all'][0][st].size}->{res['disjoint'][0][st].size}"
                                                 f" (all wake {res['disjoint_allwake'][0][st].size})"
                                                 for st in states), flush=True)
    # state blocks in STATES order, sessions in glob order within each (matches the shipped CSV's row order)
    tables = {arm: pd.concat([pd.concat(parts[arm][st], ignore_index=True) for st in states], ignore_index=True)
              for arm in parts}
    return tables["all"], tables["disjoint"], tables["disjoint_allwake"], pd.DataFrame(log)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows-csv", default=None, help="write the per-session rows-kept table here")
    ap.add_argument("--ed1f-only", action="store_true", help="skip ED2a (its table is not rewritten)")
    args = ap.parse_args()

    a_log = pd.DataFrame()
    if not args.ed1f_only:
        print("ED2a: d2 (2024 Euclidean, AR(20), 30 s / 2 s) relative to pre-task sleep", flush=True)
        a_all, a_dis, a_log = ed2a()
        shipped = pd.read_csv(ED2A_SHIPPED, float_precision="round_trip")
        dmax = assert_same_table(a_all, shipped, ["file", "source", "phase"], ["d2_rel"], ED2A_NAME)
        print(f"  all-window rebuild == shipped {ED2A_NAME}: {len(a_all)} rows, "
              f"max |d2_rel diff| = {dmax['d2_rel']:.3g}")
        a_dis = a_dis[list(shipped.columns)]
        out = os.path.join(OUT_DIR, ED2A_NAME)
        write_csv(a_dis, out)
        print(f"  disjoint: {len(a_all)} -> {len(a_dis)} rows ({a_dis['file'].nunique()} files; "
              f"dropped {sorted(DROP_DISJOINT)}) -> {out}", flush=True)

    print("ED1f: branching ratio (10 s / 2 s windows) by state", flush=True)
    b_all, b_dis, b_wake, b_log = ed1f()
    shipped = pd.read_csv(ED1F_SHIPPED, float_precision="round_trip")
    dmax = assert_same_table(b_all, shipped, ["state"], ["br", "weight"], ED1F_NAME)
    print(f"  all-window rebuild == shipped {ED1F_NAME}: {len(b_all)} rows, max |br diff| = {dmax['br']:.3g}, "
          f"max |weight diff| = {dmax['weight']:.3g}")
    b_dis = b_dis[list(shipped.columns)]
    out = os.path.join(OUT_DIR, ED1F_NAME)
    write_csv(b_dis, out)
    print(f"  disjoint: {len(b_all)} -> {len(b_dis)} rows "
          f"({b_dis['state'].value_counts().reindex(['NREM', 'REM', 'THETA', 'nonTHETA']).to_dict()}) -> {out}",
          flush=True)
    b_wake = b_wake[list(shipped.columns)]
    out_w = os.path.join(OUT_DIR_ALLWAKE, ED1F_NAME)
    write_csv(b_wake, out_w)
    print(f"  disjoint, all wake: {len(b_wake)} rows "
          f"({b_wake['state'].value_counts().reindex(['NREM', 'REM', 'THETA', 'nonTHETA']).to_dict()}) -> {out_w}",
          flush=True)
    for st in ("NREM", "REM", "THETA", "nonTHETA"):
        print(f"    median m {st:9s} task wake {b_dis.loc[b_dis.state == st, 'br'].median():.3f}   "
              f"all wake {b_wake.loc[b_wake.state == st, 'br'].median():.3f}   "
              f"windows m > 1: {int((b_wake.loc[b_wake.state == st, 'br'] > 1).sum())}")

    # confirm the written files round-trip to what was built
    built_all = ([] if args.ed1f_only else [(OUT_DIR, ED2A_NAME, a_dis)]) + [(OUT_DIR, ED1F_NAME, b_dis),
                                                                               (OUT_DIR_ALLWAKE, ED1F_NAME, b_wake)]
    for d_, name, built in built_all:
        back = pd.read_csv(os.path.join(d_, name), float_precision="round_trip")
        assert back.shape == built.shape and list(back.columns) == list(built.columns)
        num = [c for c in back.columns if pd.api.types.is_float_dtype(built[c])]
        assert all(np.array_equal(back[c].to_numpy(float), built[c].to_numpy(float), equal_nan=True) for c in num)

    if args.rows_csv:
        rows = pd.concat([a_log, b_log], ignore_index=True)
        write_csv(rows, args.rows_csv)
        print(f"  rows kept per session -> {args.rows_csv}")


if __name__ == "__main__":
    main()
