"""
fig1de_nonoverlap.py -- submitted Figure 1d and 1e redrawn on non-overlapping windows, with
the figure notebooks' own code, and Fig 1e's PPP7_day14 rebuilt from its own d2 (2024 Euclidean).

The submitted panels use every window of the 30 s / 2 s sliding grid; here they use non-overlapping
windows. The Fig 1e pairs of PPP7_day14 are built from PPP7_day14's own d2 (arm B, below).

The code below executes the producer cells from the notebooks:
  1d  first_submission_figures/Fig1_D_histogram_S1_E.ipynb
        cell 0  the loader -- newest *_CA1_D2_time_resolved_error_bar.mat per Fig-1 session, wake
                split into THETA / nonTHETA by the state at the window centre, epochs annotated;
        cell 1  plot_density_split (KDE + boxplots; wake = env != sleep) and fit_state_mixedlm, the
                window-level LMM d2 ~ C(state), groups animal, vc animal_day.
  1e  session_tables/generate_pairs_df.ipynb cells 2, 5, 13, per session -- d2 from the same
        _error_bar.mat, MUA = all CA1 spikes in 40 ms bins / n cells, 30 s centred mean sampled at
        each window centre, whole-window state guards, theta eroded / dilated by 15 s; then
        first_submission_figures/Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb cell 1
        (functions) and cell 2 (the panel: per-file d2 min-max into 6 bins, MUA dynamic range
        10 log10(p95/p05) per bin, mean +/- SEM over files; per-file Spearman + Stouffer).
  Save paths are this script's (new files only); no other report or data location is touched.
Non-overlapping = `select_disjoint`: within each session, windows of every state in time order, one
kept only if its centre is >= 30 s after the last kept one, so no two kept windows share any data;
values are the published ones, subsetted, not recomputed.
  A second, coarser selection (`fig1_nonoverlap_contrasts.select_nonoverlap`, every 15th window of
  each contiguous same-state run, arm C below) is NOT disjoint for wake: with centre labelling, theta /
  non-theta runs are short and each run restarts the count, so a sizeable fraction of its THETA and
  nonTHETA windows lie < 30 s from another kept window of the same state (NREM, REM: none).
Arms: A stored inputs, all windows; B rebuilt inputs, all windows (1e: PPP7_day14 rebuilt from its own
d2; PPP7_day7 rebuilt with every theta bout, matching the other sessions' treatment); C the
every-15th-window selection; D disjoint (the panels). 1d also gets the paired session-level contrasts
and the paper's ranked session LMM. REUSE_PAIRS=1 reads fig1e_pairs_recomputed.csv instead of
rebuilding from spikes.

    -> reports/revision_figs/fig1d_nonoverlap{,_boxplots}.{svg,png}
    -> reports/revision_figs/fig1e_nonoverlap_{NREM_vs_REM,THETA_vs_nonTHETA}_dyn_by_d2_minmax_norm.{svg,png}
    -> reports/revision_figs/fig1e_scatter_d2_mua_nonoverlap.{svg,png}  (Fig1_E cell 3, figS1_D_MUA_d2_scatter:
       d2 vs 30 s MUA, z within session x Wake/Sleep, T-maze wake theta / non-theta and NREM / REM;
       pooled r as the notebook prints it, plus per-session r with the session as the unit)
    -> data/revision/log/fig1de_nonoverlap_stats.csv, fig1e_pairs_recomputed.csv

    ENV=C:/Users/Cornell/.conda/envs/criticality/python.exe
    $ENV fig1de_nonoverlap.py
"""
from __future__ import annotations

import os
import sys
import json
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig1_nonoverlap_contrasts import contrasts, paper_test, select_nonoverlap  # noqa: E402
from run_fig1_metrics import assert_local_path  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
NB_1D = os.path.join(REPO, "first_submission_figures", "Fig1_D_histogram_S1_E.ipynb")
NB_1E = os.path.join(REPO, "first_submission_figures", "Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb")
NB_PAIRS = os.path.join(REPO, "session_tables", "generate_pairs_df.ipynb")
PAIRS_DIR = os.path.join(REPO, "data", "cross_state", "pairs")
STATES_1E = {"NREM": {"NREM"}, "REM": {"REM", "REM4"}, "THETA": {"THETA", "Wake θ"},
             "non-THETA": {"nonTHETA", "Wake non-θ"}}
NONOVL_NOTE = ("disjoint windows: per session, a window is kept only if its centre is >= 30 s from the last kept "
               "window of any state (no two share data)")
WIN_S = 30.0


def select_disjoint(W, key="animal_day", time_col="center_s"):
    """Greedy in time over every state of a session: keep a window if its centre is >= WIN_S after the
    last kept one. Window contents are then pairwise disjoint within the session."""
    W = W.sort_values([key, time_col]).reset_index(drop=True)
    t = W[time_col].to_numpy(float)
    keep = np.zeros(len(W), bool)
    for idx in W.groupby(key, sort=False).indices.values():
        last = -np.inf
        for j in idx:
            if t[j] >= last + WIN_S - 1e-6:
                keep[j], last = True, t[j]
    return W[keep]


def overlap_share(W, key="animal_day", time_col="center_s", by=None):
    """Share of windows lying < WIN_S from another window of the same session (and of `by`, if given)."""
    near = pd.Series(False, index=W.index)
    for _, g in W.groupby([key] + ([by] if by else [])):
        g = g.sort_values(time_col)
        d = np.diff(g[time_col].to_numpy(float)) < WIN_S - 1e-6
        f = np.zeros(len(g), bool)
        f[1:] |= d
        f[:-1] |= d
        near.loc[g.index] = f
    return float(near.mean())


def cell(nb, i, upto=None):
    src = "".join(json.load(open(nb, encoding="utf-8"))["cells"][i]["source"])
    if upto is not None:
        assert src.count(upto) >= 1, f"marker {upto!r} not in {os.path.basename(nb)} cell {i}"
        src = src[:src.index(upto)]
    return src


def patched(src, old, new, n):
    assert src.count(old) == n, f"expected {n} x {old!r}, found {src.count(old)}"
    return src.replace(old, new)


def save_both(path, **kw):
    """svg as the notebook saves it, plus a png; footer names the window set."""
    fig = plt.gcf()
    if SAVE_NOTE["text"]:
        fig.text(0.01, -0.02, SAVE_NOTE["text"], fontsize=7, color="0.35", ha="left", va="top")
    path = assert_local_path(str(path))
    savefig_retry(fig, path, **kw)
    kw_png = {k: v for k, v in kw.items() if k != "format"}
    savefig_retry(fig, assert_local_path(os.path.splitext(path)[0] + ".png"), **{**kw_png, "dpi": 200, "bbox_inches": "tight"})
    print(f"  saved {path} (+ .png)")


SAVE_NOTE = {"text": ""}


def savefig_retry(fig, path, tries=8, **kw):
    """Retries the save for up to ~30 s in case another process holds the file open."""
    import time
    for i in range(tries):
        try:
            return fig.savefig(path, **kw)
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(4)


# ------------------------------------------------------------------ 1d
def lmm_rows(ns, df, arm):
    env = df["environment"].astype(str).str.lower()
    sleep = df[df.state.isin(["NREM", "REM"])]
    wake = df[df.state.isin(["THETA", "nonTHETA"]) & (env != "sleep")]
    out = []
    for d, keep, lab, term in ((sleep, ["NREM", "REM"], "Sleep (NREM vs REM)", "C(state)[T.REM]"),
                               (wake, ["THETA", "nonTHETA"], "Wake (THETA vs nonTHETA)", "C(state)[T.nonTHETA]")):
        r = ns["fit_state_mixedlm"](d, keep, label=lab, reml=False)["model"]
        out.append(dict(panel="1d", arm=arm, test="window LMM d2 ~ C(state), groups animal, vc animal_day",
                        contrast=lab, n_windows=int(r.nobs), n_sessions=int(d.animal_day.nunique() if "animal_day" in d else
                                                                              (d.animal.astype(str) + d.day.astype(str)).nunique()),
                        beta=float(r.fe_params[term]), se=float(r.bse_fe[term]), p=float(r.pvalues[term])))
        print(f"  {arm:22s} {lab:26s} n={int(r.nobs):6d}  beta={r.fe_params[term]:+.4f}  SE={r.bse_fe[term]:.4f}  p={r.pvalues[term]:.3g}")
    return out


def run_1d():
    print("=== Fig 1d: notebook loader (cell 0) ===", flush=True)
    ns = {"__name__": "fig1d"}
    exec(cell(NB_1D, 0), ns)                                           # reads W: only
    c1 = cell(NB_1D, 1, upto="df_30s_only = df_binned")
    c1 = patched(c1, "plt.savefig(", "save_both(", 2)
    ns["save_both"] = save_both
    exec(c1, ns)
    A = ns["df_binned"][ns["df_binned"]["window"] == "30s"].copy()
    A["animal_day"] = A["animal"].astype(str) + "_day" + A["day"].astype(str)
    A["group"] = A["state"].astype(str)
    print(f"  all windows: {len(A)}, sessions {A.animal_day.nunique()}; per state "
          + ", ".join(f"{s}={int((A.group == s).sum())}" for s in ("NREM", "REM", "THETA", "nonTHETA")))
    A4 = A[A.group.isin(["NREM", "REM", "THETA", "nonTHETA"])]
    C = select_nonoverlap(A4)
    D = select_disjoint(A4)
    for lab, X in (("C every 15th per run (STATUS 3.5)", C), ("D disjoint", D)):
        print(f"  {lab}: {len(X)} ({len(X) / len(A):.1%}); per state "
              + ", ".join(f"{s}={int((X.group == s).sum())}" for s in ("NREM", "REM", "THETA", "nonTHETA"))
              + f" | windows < 30 s from another: same state {overlap_share(X, by='group'):.1%}, any state "
              f"{overlap_share(X):.1%}", flush=True)
    assert overlap_share(D) == 0.0

    rows = []
    print("\n  window LMM (the panel's own test; published: REM -0.0139 p 4.17e-246 n 85 829; "
          "nonTHETA +0.0099 p 3.76e-173 n 34 436)")
    rows += lmm_rows(ns, A, "A_stored_all")
    rows += lmm_rows(ns, C, "C_every15th_runs")
    rows += lmm_rows(ns, D, "D_disjoint")
    for W, arm in ((A, "A_stored_all"), (C, "C_every15th_runs"), (D, "D_disjoint")):
        W = W.copy()
        W["simplified_group"] = W.group.map({"NREM": "Sleep", "REM": "Sleep", "THETA": "Wake", "nonTHETA": "Wake"})
        print(f"\n  session level, {arm}:")
        t = paper_test(W, f"paper's ranked session LMM ({arm})")
        rows.append(dict(panel="1d", arm=arm, test="ranked session-median LMM d2_rank ~ group, groups animal",
                         contrast="Wake - Sleep (rank)", n_sessions=t["n_sessions"], beta=t["beta"], se=t["se"], p=t["p"]))
        for r in contrasts(W, arm):
            rows.append(dict(panel="1d", arm=arm, test="paired session medians, signed-rank", contrast=r["contrast"],
                             n_sessions=r["n"], median_diff=r["median"], n_pos=r["pos"], p=r["p"], cohen_d=r["cohen_d"]))

    print("\n=== Fig 1d drawn on the disjoint windows ===", flush=True)
    plt.close("all")
    SAVE_NOTE["text"] = (f"Fig 1d, d2 (2024 Euclidean, p = 20, 40 ms), 18 sessions; {NONOVL_NOTE}; "
                         + ", ".join(f"{s} n = {int((D.group == s).sum())}" for s in ("NREM", "REM", "THETA", "nonTHETA")))
    ns["plot_density_split"](D, by_window=False, title=None,
                             save_path=os.path.join(FIGDIR, "fig1d_nonoverlap.svg"), bw_adjust=0.5)
    plt.close("all")
    SAVE_NOTE["text"] = ""
    return rows


# ------------------------------------------------------------------ 1e
def recompute_pairs():
    """generate_pairs_df.ipynb, per Fig-1 session; returns the pairs table and a validation table."""
    import nelpy as nel
    from neuro_py.io import loading
    ns = {"__name__": "pairs", "plt": plt}
    exec(cell(NB_PAIRS, 2, upto="# Test the complete loading function"), ns)
    exec(cell(NB_PAIRS, 5), ns)
    exec(cell(NB_PAIRS, 13, upto="# -------- run (CA1 example) --------"), ns)
    out, val = [], []
    for bp in PATHS_1D:
        animal, day = bp.split("\\")[-2], bp.split("\\")[-1]
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        theta = nel.EpochArray(loading.load_SleepState_states(bp)["THETA"])
        ca1_df, _ = ns["load_d2_timeline_complete"](bp, "CA1")
        spike_times = st.time
        tb, fr = ns["bin_spikes"](spike_times, bin_size=0.04)
        _, P = ns["correlate_d2_mua_by_state_theta"](ca1_df, tb, fr / len(spike_times), theta,
                                                    prefer_rem_label="REM", make_scatter=False)
        P["animal"], P["day"], P["region"] = animal, day, "CA1"
        P["animal_day"] = P["animal"] + "_" + P["day"]
        f = os.path.join(PAIRS_DIR, f"{animal}_{day}_CA1_d2_mua_pairs.pkl")
        S = pd.read_pickle(f) if os.path.exists(f) else None
        v = dict(session=f"{animal}_{day}", n_recomputed=len(P), n_stored=len(S) if S is not None else 0)
        if S is not None and len(S) == len(P):
            a, b = P.sort_values(["group", "time"]).reset_index(drop=True), S.sort_values(["group", "time"]).reset_index(drop=True)
            v.update(same_rows=bool((a.group.values == b.group.values).all() and np.allclose(a.time, b.time)),
                     d2_maxdiff=float(np.nanmax(np.abs(a.d2 - b.d2))), mua_maxdiff=float(np.nanmax(np.abs(a.mua30 - b.mua30))))
        print(f"  {v}", flush=True)
        val.append(v)
        out.append(P)
    return pd.concat(out, ignore_index=True), pd.DataFrame(val)


def stouffer_rows(ns, pairs, arm):
    df = pairs.rename(columns={"mua30": "MUA"}).copy()
    df["__file__"] = df["animal"].astype(str) + "_" + df["day"].astype(str)
    rows = []
    for label, groups in STATES_1E.items():
        d = df.loc[df["group"].isin(groups), ["d2", "MUA", "__file__"]].dropna()
        res = ns["spearman_stouffer"](ns["perfile_dyn_by_bin_no_baseline"](d, method="minmax"), alternative="decreasing")
        r = dict(panel="1e", arm=arm, test="per-file Spearman(d2 bin, MUA dyn range) + Stouffer, one-sided decreasing",
                 contrast=label, n_windows=len(d), n_sessions=d["__file__"].nunique())
        if res is not None:
            r.update(z_comb=res["z_comb"], p=res["p_one_sided"], rho_wmean_fisher=res["rho_wmean_fisher"],
                     rho_median=res["rho_median"], n_files_tested=res["n_files"],
                     n_neg=int((res["per_file"]["rho"] < 0).sum()))
        print(f"  {arm:22s} {label:9s} rows={len(d):6d}  " + (f"rho_fisher={r['rho_wmean_fisher']:+.3f}  z={r['z_comb']:+.2f}  "
              f"p={r['p']:.3g}  files={r['n_files_tested']} ({r['n_neg']} neg)" if res is not None else "not testable"))
        rows.append(r)
    return rows


def run_1e():
    print("\n=== Fig 1e: generate_pairs_df rerun per session ===", flush=True)
    fpairs = os.path.join(LOGDIR, "fig1e_pairs_recomputed.csv")
    if os.environ.get("REUSE_PAIRS") == "1" and os.path.exists(fpairs):
        R, val = pd.read_csv(fpairs), None
        print(f"  REUSE_PAIRS=1: read {fpairs} ({len(R)} rows); validation table left as written", flush=True)
    else:
        R, val = recompute_pairs()
        R.to_csv(assert_local_path(fpairs), index=False)
    stored = pd.concat([pd.read_pickle(f) for f in sorted(
        os.path.join(PAIRS_DIR, x) for x in os.listdir(PAIRS_DIR) if x.endswith("_d2_mua_pairs.pkl"))], ignore_index=True)
    import glob
    import re
    from scipy import stats
    ns = {"__name__": "fig1e", "np": np, "pd": pd, "plt": plt, "stats": stats, "os": os, "re": re, "glob": glob}
    exec(cell(NB_1E, 1, upto="df_base = all_pairs.copy()"), ns)          # cell 0's imports, then cell 1's functions
    rows = []
    print("\n  monotonicity (published: NREM z -2.655 p 3.97e-3 files 17; REM -3.747 8.95e-5 14; "
          "THETA -6.533 3.22e-11 15; non-THETA -8.276 6.39e-17 17)")
    rows += stouffer_rows(ns, stored, "A_stored_all")
    rows += stouffer_rows(ns, R, "B_corrected_all")
    Rc = R.rename(columns={"time": "center_s"})
    C = select_nonoverlap(Rc)
    N = select_disjoint(Rc)
    for lab, X in (("C every 15th per run (STATUS 3.5)", C), ("D disjoint", N)):
        print(f"  {lab}: {len(X)} of {len(R)}; per group " + ", ".join(
            f"{g}={int((X.group == g).sum())}" for g in sorted(X.group.unique()))
            + f" | < 30 s from another: same group {overlap_share(X, by='group'):.1%}, any {overlap_share(X):.1%}")
    assert overlap_share(N) == 0.0
    rows += stouffer_rows(ns, C.rename(columns={"center_s": "time"}), "C_every15th_runs")
    N = N.rename(columns={"center_s": "time"})
    rows += stouffer_rows(ns, N, "D_disjoint")

    print("\n=== Fig 1e drawn on the disjoint windows (notebook cell 2) ===", flush=True)
    c2 = cell(NB_1E, 2)
    c2 = patched(c2, r'rf"D:\Documents\GitHub\Sleep_criticaltiy\reports\new_planc\individual_fig_main\fig1_E_',
                 'FIGDIR_PREFIX + rf"', 2)
    c2 = patched(c2, "plt.savefig(save_path,", "save_both(save_path,", 2)
    # REM is not drawn: on disjoint windows only 4 sessions have >= 3 d2 bins with >= 2 windows;
    # its Stouffer statistic stays in the stats table.
    c2 = patched(c2, 'for label in ["NREM", "REM"]:', 'for label in ["NREM"]:', 1)
    ns.update(all_pairs=N, FIGDIR_PREFIX=os.path.join(FIGDIR, "fig1e_nonoverlap_").replace("\\", "/"), save_both=save_both)
    plt.close("all")
    SAVE_NOTE["text"] = (f"Fig 1e, d2 (2024 Euclidean, p = 20, 40 ms), 18 sessions (PPP7_day14 rebuilt from its own d2, "
                         f"PPP7_day7 with every theta bout); {NONOVL_NOTE}; REM not drawn (too few disjoint windows per "
                         f"d2 bin: 4 sessions testable); "
                         + ", ".join(f"{g} n = {int((N.group == g).sum())}" for g in ("NREM", "REM", "Wake θ", "Wake non-θ")))
    exec(c2, ns)
    plt.close("all")
    SAVE_NOTE["text"] = ""
    rows += run_1e_scatter(stored, N)
    return rows, val


SCATTER_STATES = (("Wake non-θ", "task wake", {"Wake non-θ"}), ("Wake θ", "task wake", {"Wake θ"}),
                  ("NREM", "sleep", {"NREM"}), ("REM", "sleep", {"REM", "REM4"}))


def tag_task_epochs(pairs):
    """Fig1_E cell 0's own annotation (wake rows inside T-maze epochs get `epoch_name`)."""
    c0 = cell(NB_1E, 0)
    ns = {"__name__": "fig1e_c0", "all_pairs": pairs.copy()}
    exec(c0[c0.index("from pathlib import Path"):], ns)                  # reads W: epochs only
    return ns["all_pairs"]


def scatter_stats(ns, arm):
    """The cell's pooled r per state, plus per-session r (session as the unit)."""
    rows = []
    for st, side, labels in SCATTER_STATES:
        d = (ns["task_wake"] if side == "task wake" else ns["sleep_pairs"])
        d = d[d["group"].isin(labels)].dropna(subset=["d2_z", "mua30_z"])
        r, p = (stats_pearson(d.mua30_z, d.d2_z) if len(d) >= 3 else (np.nan, np.nan))
        per = [stats_pearson(g.mua30_z, g.d2_z)[0] for _, g in d.groupby("animal_day") if len(g) >= 5]
        per = np.array([x for x in per if np.isfinite(x)])
        from scipy.stats import wilcoxon
        zbar = float(np.tanh(np.arctanh(np.clip(per, -0.999999, 0.999999)).mean())) if per.size else np.nan
        pw = float(wilcoxon(per).pvalue) if per.size >= 3 else np.nan
        rows.append(dict(panel="1e_scatter", arm=arm, test="pooled Pearson r(MUA_z, d2_z); per-session r, Fisher mean, signed-rank",
                         contrast=f"{side}: {st}", n_windows=len(d), n_sessions=int(d.animal_day.nunique()),
                         r_pooled=r, p=p, r_session_fisher=zbar, n_pos=int((per > 0).sum()), n_sessions_r=int(per.size),
                         p_session=pw))
        print(f"  {arm:22s} {side:9s} {st:10s} n={len(d):6d}  pooled r={r:+.4f} p={p:.3g} | per session (>= 5 windows) "
              f"{int((per > 0).sum())}/{per.size} positive, Fisher mean {zbar:+.3f}, signed-rank p={pw:.3g}")
    return rows


def stats_pearson(x, y):
    from scipy.stats import pearsonr
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return pearsonr(x[ok], y[ok]) if ok.sum() >= 3 else (np.nan, np.nan)


def run_1e_scatter(stored, N):
    """Fig1_E cell 3 (figS1_D_MUA_d2_scatter): A on the stored pairs, D on the rebuilt,
    disjoint pairs (the panel)."""
    print("\n=== Fig 1e scatter (notebook cell 3): d2_z vs MUA_z, z within session x Wake/Sleep ===", flush=True)
    print("  published: Wake non-θ n 9 078 r -0.5248 | Wake θ n 1 372 r +0.1495 | NREM n 78 517 r +0.2253 | REM n 1 312 r +0.1210")
    c3 = cell(NB_1E, 3)
    c3 = patched(c3, r'fig.savefig(r"D:\Documents\GitHub\Sleep_criticaltiy\reports\new_planc\supplementary\individual_fig_supp\figS1_D_MUA_d2_scatter.svg",',
                 "save_both(SCATTER_OUT,", 1)
    rows = []
    for pairs, arm, draw in ((stored, "A_stored_all", False), (N, "D_disjoint", True)):
        ns = {"__name__": "fig1e_c3", "all_pairs": tag_task_epochs(pairs), "save_both": save_both,
              "SCATTER_OUT": os.path.join(FIGDIR, "fig1e_scatter_d2_mua_nonoverlap.svg")}
        plt.close("all")
        SAVE_NOTE["text"] = (f"Fig 1e scatter, d2 (2024 Euclidean, p = 20, 40 ms, full window) vs 30 s MUA, z within session x "
                             f"Wake/Sleep, 18 sessions; {NONOVL_NOTE}") if draw else ""
        src = c3 if draw else c3.replace("save_both(SCATTER_OUT,", "(lambda *a, **k: None)(SCATTER_OUT,")
        exec(src, ns)
        rows += scatter_stats(ns, arm)
        plt.close("all")
    SAVE_NOTE["text"] = ""
    return rows


def main():
    global PATHS_1D
    src0 = cell(NB_1D, 0)
    ns0 = {}
    exec(src0[src0.index("paths = ["):src0.index("regions = ")], ns0)    # the notebook's 18-session list
    PATHS_1D = ns0["paths"]
    rows = run_1d()
    r1e, val = run_1e()
    rows += r1e
    S = pd.DataFrame(rows)
    S.to_csv(assert_local_path(os.path.join(LOGDIR, "fig1de_nonoverlap_stats.csv")), index=False)
    if val is not None:
        val.to_csv(assert_local_path(os.path.join(LOGDIR, "fig1e_pairs_validation.csv")), index=False)
    pd.set_option("display.width", 250)
    print("\n=== summary ===")
    print(S.drop(columns=["test"]).to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    print("\nwrote fig1de_nonoverlap_stats.csv, fig1e_pairs_recomputed.csv, fig1e_pairs_validation.csv")


PATHS_1D: list = []

if __name__ == "__main__":
    main()
