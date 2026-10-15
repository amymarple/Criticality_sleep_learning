"""
fig1_d2_vs_mua_2x2.py — d2 vs MUA, per state, across the definition x excision 2x2.

This is a **Figure 1** analysis (cross-state metric behaviour), not Figure 2: it is the
revision counterpart of `figures/Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb`
and it uses the Fig-1 18-session set wherever the data exist.

Four panels:
    Euclidean full   |  Euclidean ripple-excised
    KL full          |  KL ripple-excised

with one colour per state and a separate least-squares fit line per state, using the
colour scheme from `Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb` cell 2:

    NREM #7b3294   REM #d7b5d8   THETA #d95f0e   non-THETA #fdb863

> **The ablated panels are NREM-only.** `helper/time_resolved_d2_across_state_ripple_ablate.m`
> guards the ripple subtraction with `isNREM`, so REM and WAKE rows in
> `*_rippleablate.mat` are byte-identical to the unablated file — there is no ablated
> REM/THETA to plot. The KL-excised arm (`r35_kl_rippleablate.py`) was likewise computed
> for NREM only.

The 2x2 compares the d2-MUA relationship across the Euclidean and KL definitions,
full-window and ripple-excised, by state.

Sources, all already computed:
    Euclidean full      `d2` in data/revision/*_fig1_metrics_nonoverlap.pkl (joined from
                        the MATLAB *_D2_time_resolved.mat at 100% of windows)
    Euclidean excised   *_CA1_D2_time_resolved_rippleablate.mat, joined on window centre
    KL full             kl_error_bars.csv (NREM) + kl_error_bars_states.csv (other states)
    KL excised          r35_kl_rippleablate.csv (NREM)

Windows are the revision pipeline's NON-OVERLAPPING NREM/REM/THETA/nonTHETA tiling.
Full-window arms: all 18 Fig-1 sessions. Excised arms: the 10 with a
rippleablate .mat. n is stated per arm in the legend.

    -> reports/revision_figs/fig1_d2_vs_mua_2x2.png / .svg
    -> data/revision/log/fig1_d2_vs_mua_2x2.csv
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
REVDIR = os.path.join(REPO, "data", "revision")
LOGDIR = os.path.join(REVDIR, "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")

# exactly the notebook's palette (cell 2)
COLORS = {"NREM": "#7b3294", "REM": "#d7b5d8",
          "THETA": "#d95f0e", "nonTHETA": "#fdb863"}
STATE_ORDER = ["NREM", "REM", "THETA", "nonTHETA"]

# The Fig-1 set: 18 sessions, 6 mice, CA1. The FULL-window arms use all 18. The
# ripple-excised arms are limited to the 10 sessions that have a
# *_D2_time_resolved_rippleablate.mat on the share (PPP7 day12/14, PPP8 day14,
# PPP20 day7/8/9, PPP21 day7/8 do not) — n is reported per arm rather than silently
# equalised by dropping sessions from the full arms.
SESSIONS = {f"{a}_{d}": rf"W:\data\PPP\{a}\{d}" for a, d in [
    ("PPP4", "day11"),
    ("PPP7", "day7"), ("PPP7", "day10"), ("PPP7", "day12"), ("PPP7", "day14"),
    ("PPP8", "day8"), ("PPP8", "day14"),
    ("PPP20", "day4"), ("PPP20", "day5"), ("PPP20", "day7"), ("PPP20", "day8"),
    ("PPP20", "day9"),
    ("PPP21", "day4"), ("PPP21", "day5"), ("PPP21", "day7"), ("PPP21", "day8"),
    ("PPP23", "day7"), ("PPP23", "day8"),
]}

PANELS = [
    ("d2_eu",      "d2 (2024 Euclidean, p=20)\nfull window",      False),
    ("d2_eu_abl",  "d2 (2024 Euclidean, p=20)\nRIPPLE TIME EXCISED", True),
    ("d2_kl",      "d2 (2025 KL, p=20)\nfull window",             False),
    ("d2_kl_abl",  "d2 (2025 KL, p=20)\nRIPPLE TIME EXCISED",     True),
]


def assert_local_path(path):
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    if os.path.splitdrive(p)[0].rstrip(":").upper() not in {"C", "D"}:
        raise PermissionError(f"refusing to write outside C:/D:: {p}")
    return p


def nrem_mat(basepath, tag):
    day = os.path.basename(basepath.rstrip("\\/"))
    f = os.path.join(basepath, "complex_system",
                     f"{day}_CA1_D2_time_resolved{tag}.mat")
    if not os.path.exists(f):
        return None
    m = loadmat(f, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    out = []
    for r in m["results"]:
        if "NREM" not in str(r["state"]).upper():
            continue
        out.append(pd.DataFrame({"center_s": np.atleast_1d(r["center"]).astype(float),
                                 "v": np.atleast_1d(r["d2"]).astype(float)}))
    return pd.concat(out).sort_values("center_s").reset_index(drop=True) if out else None


def build():
    KLn = pd.read_csv(os.path.join(LOGDIR, "kl_error_bars.csv"))
    KLn["state"] = "NREM"
    KLs = pd.read_csv(os.path.join(LOGDIR, "kl_error_bars_states.csv"))
    KL = pd.concat([KLn, KLs], ignore_index=True)
    KL["session"] = KL.animal.astype(str) + "_" + KL.day.astype(str)
    KL = KL[["session", "state", "w", "d2mle_p20"]].rename(
        columns={"d2mle_p20": "d2_kl"})

    KA = pd.read_csv(os.path.join(LOGDIR, "r35_kl_rippleablate.csv"))
    if "session" not in KA.columns:
        KA["session"] = KA.animal.astype(str) + "_" + KA.day.astype(str)
    KA["state"] = "NREM"
    KA = KA[["session", "state", "w", "kl_abl"]].rename(columns={"kl_abl": "d2_kl_abl"})

    parts = []
    for sess, bp in SESSIONS.items():
        f = os.path.join(REVDIR, f"{sess}_CA1_fig1_metrics_nonoverlap.pkl")
        if not os.path.exists(f):
            print(f"  [skip] {sess}: no fig1 pickle")
            continue
        with open(f, "rb") as fh:
            W = pickle.load(fh)["windows"]
        W = W[W["nonoverlap"]] if "nonoverlap" in W.columns else W
        W = W[["state", "win_start", "center_s", "MUA_mean", "d2"]].copy()
        W = W.rename(columns={"d2": "d2_eu"})
        W["session"] = sess
        W["w"] = W["win_start"].round().astype(int)

        A = nrem_mat(bp, "_rippleablate")
        if A is not None:
            W = pd.merge_asof(W.sort_values("center_s"),
                              A.rename(columns={"v": "d2_eu_abl"}),
                              on="center_s", tolerance=0.6, direction="nearest")
            W.loc[W.state != "NREM", "d2_eu_abl"] = np.nan   # ablation is NREM-only
        else:
            W["d2_eu_abl"] = np.nan
        W = W.merge(KL[KL.session == sess], on=["session", "state", "w"], how="left")
        W = W.merge(KA[KA.session == sess], on=["session", "state", "w"], how="left")
        parts.append(W)
        print(f"  {sess:12} {len(W):4d} windows | "
              + "  ".join(f"{c} {int(W[c].notna().sum())}"
                          for c in ("d2_eu", "d2_eu_abl", "d2_kl", "d2_kl_abl")),
              flush=True)
    return pd.concat(parts, ignore_index=True)


def zscore_within_session(D):
    """z-score MUA and every d2 column WITHIN each session.

    Sessions differ in absolute firing rate and in absolute d2, and those offsets invert
    the pooled NREM correlation relative to the per-session one. Removing them per
    session makes pooling across sessions valid, and one fit line per state then matches
    the per-session estimate rather than reflecting a between-session artifact. Each d2
    column is standardised over the windows where that column exists (the excised
    columns are NREM-only by construction, see the header note)."""
    D = D.copy()
    for c in ["MUA_mean"] + [p[0] for p in PANELS]:
        out = np.full(len(D), np.nan)
        for _s, idx in D.groupby("session").groups.items():
            v = D.loc[idx, c].to_numpy(float)
            m = np.isfinite(v)
            if m.sum() >= 10 and np.nanstd(v[m]) > 0:
                out[D.index.get_indexer(idx)[m]] = (
                    (v[m] - np.nanmean(v[m])) / np.nanstd(v[m]))
        D[("MUA_z" if c == "MUA_mean" else c + "_z")] = out
    return D


def main():
    D = zscore_within_session(build())
    assert_local_path(LOGDIR)
    D.to_csv(assert_local_path(os.path.join(LOGDIR, "fig1_d2_vs_mua_2x2.csv")),
             index=False)

    print(f"\n{'='*88}\nd2 vs MUA — PER-SESSION Spearman (Fisher-z), pooled shown only "
          f"for contrast\n{'='*88}")
    print("  Both axes are z-scored WITHIN SESSION first, which removes the\n"
          "  between-session offsets in MUA and d2. Without that, pooling inverts the\n"
          "  NREM sign (raw pooled -0.18 vs per-session +0.22). After z-scoring the two\n"
          "  columns below agree, so the single pooled fit line per state is honest.\n")
    print(f"{'panel':34}" + "".join(f"{s:>18}" for s in STATE_ORDER))
    print(f"{'':34}" + "".join(f"{'per-sess (pooled)':>18}" for _ in STATE_ORDER))
    rows = []
    for col, lbl, _ in PANELS:
        cells = ""
        for st in STATE_ORDER:
            g = D[(D.state == st)][["session", col + "_z", "MUA_z"]].dropna()
            if len(g) >= 20:
                pooled = stats.spearmanr(g["MUA_z"], g[col + "_z"]).statistic
                per = [stats.spearmanr(x["MUA_z"], x[col + "_z"]).statistic
                       for _s, x in g.groupby("session") if len(x) >= 10]
                per = np.asarray([v for v in per if np.isfinite(v)], float)
                z = np.arctanh(np.clip(per, -0.999999, 0.999999))
                rbar = float(np.tanh(z.mean())) if z.size else np.nan
                se = (float(z.std(ddof=1) / np.sqrt(z.size)) if z.size > 1 else np.nan)
                cells += f"{rbar:+.2f} ({pooled:+.2f}) n{z.size:<3}".rjust(18)
                rows.append(dict(panel=col, state=st, n_win=len(g), n_sess=int(z.size),
                                 r_per_session=rbar, se=se, r_pooled=pooled))
            else:
                cells += f"{'-':>18}"
        print(f"{lbl.splitlines()[0][:20]:22}{lbl.splitlines()[1][:11]:12}{cells}")
    pd.DataFrame(rows).to_csv(assert_local_path(os.path.join(
        LOGDIR, "fig1_d2_vs_mua_2x2_stats.csv")), index=False)
    plot(D)


def plot(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({
        "svg.fonttype": "none", "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "axes.unicode_minus": False, "axes.linewidth": 1.2,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
    })
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 8.4))
    for ax, (col, lbl, nrem_only) in zip(axes.ravel(), PANELS):
        states = ["NREM"] if nrem_only else STATE_ORDER
        for st in states:
            g = D[D.state == st][["session", col + "_z", "MUA_z"]].dropna()
            if len(g) < 20:
                continue
            x, y = g["MUA_z"].to_numpy(float), g[col + "_z"].to_numpy(float)
            ax.scatter(x, y, s=5, alpha=0.13, color=COLORS[st], linewidths=0,
                       rasterized=True)
            # One pooled fit per state: both axes are z-scored WITHIN SESSION first,
            # which removes the between-session offsets in MUA and d2 that can
            # otherwise invert the pooled sign relative to the per-session estimate
            # when windows are pooled (a Simpson's paradox). After within-session
            # z-scoring, the pooled estimate agrees with the per-session one.
            b = np.polyfit(x, y, 1)
            xs = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 50)
            r = stats.spearmanr(x, y).statistic
            n_s = g["session"].nunique()
            ax.plot(xs, np.polyval(b, xs), color=COLORS[st], lw=2.6,
                    label=f"{st}  r={r:+.2f}  ({n_s} sess, {len(g)} win)")
        ax.set_xlabel("MUA (z within session)")
        ax.set_ylabel("d2 (z within session)")
        ax.set_title(lbl, fontsize=10)
        ax.legend(frameon=False, fontsize=8.5, loc="upper right")
        if nrem_only:
            ax.text(0.98, 0.02, "ablation is NREM-only\n(isNREM guard in the producer)",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=7.5, color="0.45")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    assert_local_path(FIGDIR)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(
            FIGDIR, f"fig1_d2_vs_mua_2x2.{ext}")), dpi=300, bbox_inches="tight")
    print(f"\nfigure -> {FIGDIR}\\fig1_d2_vs_mua_2x2.png/.svg")


if __name__ == "__main__":
    main()
