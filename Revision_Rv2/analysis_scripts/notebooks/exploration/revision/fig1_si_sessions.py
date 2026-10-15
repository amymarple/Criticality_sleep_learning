"""
fig1_si_sessions.py -- structure index of d2 per session against the session's own null, in the format
of the lab's pupil paper (two columns per session: the index and its null).

The published format (pupil_project/misc/Figure1/SleepState_structureIndex_stats_Fig1e.m) collects, per
session, [Structure_index, prctile(shuf_SI_vals, 95)]: the PridaLab structure index (SI) of the label in
the raw feature space over all states (n_neighbors 20; SleepState_rawLFPfeatures_ALLStates_StructureIndex_
Fig1e.m) and the 95th percentile of its shuffle null. One number and its null per session: two columns.

Here: label d2 (2024 Euclidean, 40 ms, AR p = 20), raw (one session at a time). Feature space: the 10
window features z-scored per session, every non-overlapping 30 s window of the session, all states
(cohort and missing-value rule of fig1_nrem_feature_gradient: the Fig-1 sessions with SWR detection). SI:
10 bins, n_neighbors 20, as the pupil paper's all-state SI. Null (drawn): time-preserving -- d2 and the
features are autocorrelated in time, so d2 is circularly shifted within each state episode (`iv_idx`,
the scored interval the window was cut from) by 1..L-1 windows rather than label-shuffled; a one-window
episode keeps its own d2 value unchanged. Because every episode keeps its own d2 values, this null
also keeps the d2 difference BETWEEN states and between episodes: the session's SI has to exceed what
state and episode identity alone give. The pupil paper's own null (label shuffles) is computed on the
same windows and kept in the table (`shuffle_p95`), not drawn. N_NULL surrogates for both.
Across sessions: SI above its own null 95th; Wilcoxon signed-rank of SI - null 95th; per-mouse means.

Writes only to D:.
    -> data/revision/log/fig1_si_sessions.csv
    -> reports/revision_figs/fig1_si_sessions.png / .svg
"""
from __future__ import annotations

import os
import sys
import zlib

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                        # noqa: E402
from fig1_state_space_umap import FEATURES, LOGDIR, FIGDIR            # noqa: E402
from fig1_state_feature_gradient import load                          # noqa: E402
from fig1_lfp_manifold import SI_PATH                                 # noqa: E402
from m25_tmaze_box import MOUSE_COL, MOUSE_LABEL      # noqa: E402  the manuscript's Mouse 1-6 (ID order) and colours

N_NULL = int(os.environ.get("N_NULL", "500"))
SI_BINS, SI_K = 10, 20
OFFSET = 1
OUT = os.path.join(LOGDIR, "fig1_si_sessions.csv")


def si(X, y):
    if SI_PATH not in sys.path:
        sys.path.insert(0, SI_PATH)
    from structure_index import compute_structure_index
    v = compute_structure_index(X, y.copy(), n_bins=SI_BINS, n_neighbors=SI_K, num_shuffles=0, verbose=False)[0]
    return float(v) if np.isfinite(v) else 0.0


def run_session(args):
    s, g = args
    g = g.sort_values("center_s").reset_index(drop=True)
    rng = np.random.default_rng(zlib.crc32(s.encode()))
    X, y = g[FEATURES].to_numpy(float), g.d2.to_numpy(float)
    groups = [idx for idx in (np.flatnonzero(g.ep.to_numpy() == u) for u in pd.unique(g.ep)) if len(idx) > 2 * OFFSET]
    shift = np.empty(N_NULL)
    for k in range(N_NULL):
        yn = y.copy()
        for idx in groups:
            yn[idx] = np.roll(y[idx], rng.integers(OFFSET, len(idx) - OFFSET + 1))
        shift[k] = si(X, yn)
    shuffle = np.array([si(X, rng.permutation(y)) for _ in range(N_NULL)])
    obs = si(X, y)
    print(f"  {s}: SI {obs:.3f}, null95 {np.percentile(shift, 95):.3f}", flush=True)
    return dict(session=s, animal=s.split("_")[0], n_windows=len(g), n_episodes=g.ep.nunique(),
                frac_windows_shifted=sum(len(i) for i in groups) / len(g),
                SI=obs, null_p95=np.percentile(shift, 95), null_median=np.median(shift),
                p=(1 + np.sum(shift >= obs)) / (1 + N_NULL),
                shuffle_p95=np.percentile(shuffle, 95), p_shuffle=(1 + np.sum(shuffle >= obs)) / (1 + N_NULL))


def plot(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    animals = sorted(R.animal.unique(), key=lambda a: int(a[3:]))
    acol = MOUSE_COL
    fig, ax = plt.subplots(figsize=(2.6, 3.2))
    for _, r in R.iterrows():
        ax.plot([0, 1], [r.SI, r.null_p95], color="0.75", lw=0.7, zorder=1)
        ax.scatter(0, r.SI, s=18, color=acol[r.animal], zorder=3, lw=0)
        ax.scatter(1, r.null_p95, s=18, color="0.55", zorder=3, lw=0)
    for x, col in ((0, "SI"), (1, "null_p95")):
        ax.plot([x - 0.2, x + 0.2], [R[col].median()] * 2, color="k", lw=1.8, zorder=4)
    d = (R.SI - R.null_p95).to_numpy()
    p = stats.wilcoxon(d).pvalue
    am = R.groupby("animal").apply(lambda g: (g.SI - g.null_p95).mean())
    top = max(R.SI.max(), R.null_p95.max()) * 1.3
    ax.text(0.5, top, f"{int((d > 0).sum())}/{len(R)} sessions above own null\n{int((am > 0).sum())}/{len(am)} mice; "
            f"Wilcoxon p = {p:.2g}", ha="center", va="top", fontsize=7)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["d2 SI", "null\n(95th)"]); ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, top * 1.02); ax.set_ylabel("structure index")
    h = [plt.Line2D([], [], marker="o", ls="", color=acol[a], label=MOUSE_LABEL[a]) for a in animals]
    ax.legend(handles=h, frameon=False, fontsize=6.5, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("d2 (2024 Euclidean) in the\n10-feature space, all states", loc="left", fontsize=8)
    fig.text(0.0, -0.03,
             f"Per session: PridaLab structure index ({SI_BINS} bins, {SI_K} neighbours) of d2 over every non-overlapping 30 s "
             f"window,\nfeatures z-scored within session. Null: d2 circularly shifted within each state episode ({N_NULL} "
             "surrogates; keeps the d2\ndifferences between states and episodes), 95th percentile. Black bar: median. "
             f"Pupil-paper label shuffle: {int((R.SI > R.shuffle_p95).sum())}/{len(R)} above.", fontsize=6, va="top")
    o = os.path.join(FIGDIR, "fig1_si_sessions")
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{o}.{ext}"), dpi=200, bbox_inches="tight")
    print("->", o + ".png")


def main():
    if os.environ.get("PLOT_ONLY"):
        return plot(pd.read_csv(OUT))
    Z = load()
    print(f"{Z.session.nunique()} sessions, {len(Z)} windows; {N_NULL} surrogates")
    from multiprocessing import Pool
    with Pool(min(16, os.cpu_count() or 1)) as pool:
        R = pd.DataFrame(pool.map(run_session, list(Z.groupby("session"))))
    R.to_csv(assert_local_path(OUT), index=False)
    print(f"median SI {R.SI.median():.3f}; own null95 {R.null_p95.median():.3f} -> {int((R.SI > R.null_p95).sum())}/{len(R)} above; "
          f"shuffle null95 {R.shuffle_p95.median():.3f} -> {int((R.SI > R.shuffle_p95).sum())}/{len(R)} above")
    plot(R)


if __name__ == "__main__":
    main()
