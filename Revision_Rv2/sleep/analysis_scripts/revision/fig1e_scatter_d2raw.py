"""
fig1e_scatter_d2raw.py -- the Fig 1e d2-MUA scatter (Fig1_E cell 3, figS1_D_MUA_d2_scatter) with d2 in
raw units and only MUA z-scored, on the disjoint windows of fig1de_nonoverlap.py.

The published scatter z-scores both axes within session x Wake/Sleep, which destroys d2's absolute
scale. Here d2 is kept raw and only MUA is z-scored within session. Leaving MUA raw as well would let
the between-session offsets of both axes (unit yield moves d2 and per-cell MUA) produce a Simpson
inversion when windows are pooled (opposite-signed per-session and pooled correlations). With MUA
z-scored within session every session's MUA mean is 0, so a pooled slope is a weighted mean of the
within-session slopes; d2's session offsets only widen the cloud vertically.

Windows and states exactly as fig1de_nonoverlap.py's scatter: rebuilt pairs
(`fig1e_pairs_recomputed.csv`: d2 (2024 Euclidean, p = 20, 40 ms) from each session's
_error_bar.mat, full window; MUA = all CA1 spikes / n cells, 40 ms bins, 30 s centred mean), the
disjoint selection, task wake = T-maze epochs (the notebook's cell-0 tagging) theta / non-theta, and
NREM / REM. MUA z-scored within session x Wake/Sleep as the notebook does (ddof 0, over all windows of
that session and group, before task selection). d2 untouched. Plot style copied from the cell
(colours, 4 pt points, dashed line over the state's 1-99 % MUA range) except the line: the median
within-session slope drawn through the state's centroid, not the pooled OLS (which, with MUA z-scored
over all wake, keeps between-session offsets in theta / non-theta and overstates their slopes); REM is
drawn as points without a line.
Statistics per session (>= 5 windows): Pearson r (equal to the z-scored version's per-session r),
Fisher mean, sessions in direction, signed-rank p; OLS slope in d2 per SD of MUA and the d2 change
across the session's 5-95 % MUA range. The pooled slope (the drawn line) and pooled r are reported too.

    -> reports/revision_figs/fig1e_scatter_d2raw_muaz_nonoverlap.{svg,png}
    -> data/revision/log/fig1e_scatter_d2raw_{stats,per_session}.csv   (WAKE=all: *_allwake)
    Statistics per state: the per-session signed-rank test (sessions >= 5 windows) and, weighting each session by
    its window count, a random-effects meta-analysis of the per-session Fisher z (meta_r)

    ENV=C:/Users/Cornell/.conda/envs/criticality/python.exe
    $ENV fig1de_nonoverlap.py        (once; writes fig1e_pairs_recomputed.csv)
    $ENV fig1e_scatter_d2raw.py
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.stats import pearsonr, wilcoxon  # noqa: E402

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig1de_nonoverlap import FIGDIR, LOGDIR, NONOVL_NOTE, select_disjoint, tag_task_epochs  # noqa: E402
from run_fig1_metrics import assert_local_path  # noqa: E402

# Fig1_E cell 3's palettes
WAKE_PAL = {"Wake θ": "#d95f0e", "Wake non-θ": "#fdb863"}
SLEEP_PAL = {"NREM": "#7b3294", "REM": "#d7b5d8"}
WAKE_LINE = {"Wake θ": "#97550a", "Wake non-θ": "#b87f44"}
SLEEP_LINE = {"NREM": "#56236a", "REM": "#976695"}
MIN_WIN = 5
# WAKE=all: every wake window, home cage included, matching what the dynamic range (Fig1_E cell 2)
# uses; default task = T-maze epochs only, as Fig1_E cell 3 (the published scatter).
WAKE = os.environ.get("WAKE", "task")
META_MIN_N = 4                     # the meta-analysis needs n > 3 windows (Fisher z variance 1 / (n - 3))


def safe_z(s):
    sd = s.astype(float).std(ddof=0)
    return (s - s.mean()) / sd if np.isfinite(sd) and sd > 0 else s * np.nan


def per_session(d):
    rows = []
    for s, g in d.groupby("animal_day"):
        if len(g) < MIN_WIN or g.mua_z.std() == 0:
            continue
        x, y = g.mua_z.to_numpy(float), g.d2.to_numpy(float)
        b = np.polyfit(x, y, 1)[0]
        lo, hi = np.percentile(x, [5, 95])
        rows.append(dict(session=s, n=len(g), r=pearsonr(x, y)[0], slope_per_sd=b, d2_change_5_95=b * (hi - lo),
                         d2_median=float(np.median(y))))
    return pd.DataFrame(rows)


def meta_r(d, min_n=META_MIN_N):
    """Per-session correlations combined with weights for how many windows each session has: Fisher
    z_i = arctanh(r_i) with variance 1 / (n_i - 3).
    Fixed effect: inverse-variance weighted mean of z_i. Random effects (DerSimonian-Laird, primary): weights
    1 / (v_i + tau^2), so a few large sessions cannot carry the result if sessions genuinely differ. Two-sided z
    tests; the pooled r = tanh(pooled z)."""
    from scipy.stats import norm
    rows = []
    for s, g in d.groupby("animal_day"):
        if len(g) < min_n or g.mua_z.std() == 0 or g.d2.std() == 0:
            continue
        rows.append((len(g), pearsonr(g.mua_z.to_numpy(float), g.d2.to_numpy(float))[0]))
    if len(rows) < 2:
        return {}
    n = np.array([a for a, _ in rows], float)
    y = np.arctanh(np.clip(np.array([b for _, b in rows], float), -0.999999, 0.999999))
    v = 1.0 / (n - 3)
    w = 1.0 / v
    y_fe = float(np.sum(w * y) / np.sum(w)); se_fe = float(np.sqrt(1.0 / np.sum(w)))
    k = len(y)
    Q = float(np.sum(w * (y - y_fe) ** 2))
    C = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    tau2 = max(0.0, (Q - (k - 1)) / C) if C > 0 else 0.0
    ws = 1.0 / (v + tau2)
    y_re = float(np.sum(ws * y) / np.sum(ws)); se_re = float(np.sqrt(1.0 / np.sum(ws)))
    return dict(meta_k=k, meta_windows=int(n.sum()), meta_min_n=min_n,
                r_fe=float(np.tanh(y_fe)), p_fe=float(2 * norm.sf(abs(y_fe / se_fe))),
                r_re=float(np.tanh(y_re)), r_re_lo=float(np.tanh(y_re - 1.96 * se_re)),
                r_re_hi=float(np.tanh(y_re + 1.96 * se_re)), z_re=float(y_re / se_re),
                p_re=float(2 * norm.sf(abs(y_re / se_re))), tau2=tau2, Q=Q,
                I2=float(max(0.0, (Q - (k - 1)) / Q)) if Q > 0 else 0.0)


def summarise(state, d, P):
    r = P.r.to_numpy()
    b, a = np.polyfit(d.mua_z, d.d2, 1)
    s = dict(state=state, n_windows=len(d), n_sessions_all=int(d.animal_day.nunique()), n_sessions=len(P),
             r_fisher_mean=float(np.tanh(np.arctanh(np.clip(r, -0.999999, 0.999999)).mean())),
             n_pos=int((r > 0).sum()), p_signed_rank=float(wilcoxon(r).pvalue) if len(P) >= 3 else np.nan,
             slope_per_sd_median=float(P.slope_per_sd.median()), d2_change_5_95_median=float(P.d2_change_5_95.median()),
             d2_median_of_sessions=float(P.d2_median.median()),
             slope_pooled=float(b), intercept_pooled=float(a), r_pooled=float(pearsonr(d.mua_z, d.d2)[0]))
    s.update(meta_r(d))
    print(f"  {state:10s} {len(d):5d} windows | per session (>= {MIN_WIN} windows): r Fisher mean {s['r_fisher_mean']:+.2f}, "
          f"{s['n_pos']}/{len(P)} positive, p = {s['p_signed_rank']:.3g}; median slope {s['slope_per_sd_median']:+.4f} d2 per SD, "
          f"5-95 % MUA range moves d2 by {s['d2_change_5_95_median']:+.4f} (median d2 {s['d2_median_of_sessions']:.3f}) | "
          f"pooled slope {b:+.4f}, pooled r {s['r_pooled']:+.2f}")
    if "r_re" in s:
        print(f"  {'':10s} window-weighted ({s['meta_k']} sessions with >= {META_MIN_N} windows): random effects r "
              f"{s['r_re']:+.2f} [{s['r_re_lo']:+.2f}, {s['r_re_hi']:+.2f}], p = {s['p_re']:.3g} (I2 {s['I2']:.0%}); "
              f"fixed effect r {s['r_fe']:+.2f}, p = {s['p_fe']:.3g}")
    return s


def local_limits(x, pad_frac=0.1):
    lo, hi = np.percentile(np.asarray(x, float), [1, 99])
    pad = pad_frac * (hi - lo if hi > lo else 1.0)
    return lo - pad, hi + pad


def main():
    R = pd.read_csv(os.path.join(LOGDIR, "fig1e_pairs_recomputed.csv"))
    N = select_disjoint(R.rename(columns={"time": "center_s"})).rename(columns={"center_s": "time"})
    T = tag_task_epochs(N)
    T["group2"] = np.where(T.group.isin(["Wake θ", "Wake non-θ"]), "Wake", "Sleep")
    T["mua_z"] = T.groupby(["animal_day", "group2"])["mua30"].transform(safe_z)      # MUA only; d2 stays raw
    task = WAKE != "all"
    wake = (lambda st: T[(T.group == st) & T.epoch_name.notna()]) if task else (lambda st: T[T.group == st])
    panels = [(("Task wake (T-maze)" if task else "Wake (all)") + "\nWake θ + Wake non-θ", [
                  ("Wake non-θ", wake("Wake non-θ"), WAKE_PAL, WAKE_LINE, 0.1),
                  ("Wake θ", wake("Wake θ"), WAKE_PAL, WAKE_LINE, 0.1)]),
              ("Sleep\nNREM + REM", [
                  ("NREM", T[T.group == "NREM"], SLEEP_PAL, SLEEP_LINE, 0.3),
                  ("REM", T[T.group.isin(["REM", "REM4"])], SLEEP_PAL, SLEEP_LINE, 0.05)])]
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
                         "axes.linewidth": 1.5, "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    fig, axes = plt.subplots(1, 2, figsize=(10, 6.2))
    stats, persess = [], []
    print("d2 (2024 Euclidean) raw vs MUA z (within session x Wake/Sleep), disjoint windows")
    for ax, (title, groups) in zip(axes, panels):
        handles = []
        for st, d, pal, lpal, pad in groups:
            d = d.dropna(subset=["d2", "mua_z"])
            ax.scatter(d.mua_z, d.d2, s=4, alpha=0.8, color=pal[st], edgecolors="none", rasterized=True)
            P = per_session(d)
            s = summarise(st, d, P)
            stats.append(s)
            persess.append(P.assign(state=st))
            # Line = the median WITHIN-session slope through the state's centroid (descriptive). The pooled OLS
            # line is not drawn: MUA is z-scored over all wake, so theta / non-theta keep between-session
            # offsets and the pooled slope overstates the within-session effect.
            # REM: points only (few disjoint windows per session, mixed sign across sessions), no trend line.
            if st != "REM":
                lo, hi = local_limits(d.mua_z, pad)
                xs = np.linspace(lo, hi, 200)
                cx, cy = float(d.mua_z.mean()), float(d.d2.mean())
                ax.plot(xs, cy + s["slope_per_sd_median"] * (xs - cx), color=lpal[st], linestyle="--", lw=1.8)
                handles += [Line2D([0], [0], marker="o", linestyle="None", color=pal[st], label=st),
                            Line2D([0], [0], color=pal[st], linestyle="--", lw=1.8,
                                   label=f"r = {s['r_fisher_mean']:+.2f}, {s['n_pos']}/{s['n_sessions']} sessions, "
                                         f"p = {s['p_signed_rank']:.2g}")]
            else:
                handles += [Line2D([0], [0], marker="o", linestyle="None", color=pal[st],
                                   label=f"REM ({len(d)} windows, {d.animal_day.nunique()} sessions): no line, "
                                         f"{s['n_pos']}/{s['n_sessions']} > 0")]
        ax.set_title(title)
        ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=1)
        ax.set_xlabel("Mean population firing rate (z, within session)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(False)
    fig.supylabel("d2 distance (2024 Euclidean)")
    fig.tight_layout()
    import textwrap
    note = ("Fig 1e scatter: d2 raw (p = 20, 40 ms, full window), MUA z-scored within session x Wake/Sleep; "
            f"dashed = median within-session slope through the state centroid (descriptive); r = per-session Pearson, "
            f"Fisher mean, sessions with >= {MIN_WIN} windows, signed-rank; {NONOVL_NOTE}")
    fig.text(0.01, -0.02, textwrap.fill(note, 170), fontsize=7, color="0.35", ha="left", va="top")
    sfx = "" if task else "_allwake"                    # the default (task wake) outputs keep their names
    out = os.path.join(FIGDIR, "fig1e_scatter_d2raw_muaz_nonoverlap" + sfx)
    fig.savefig(assert_local_path(out + ".svg"), dpi=300, bbox_inches="tight")
    fig.savefig(assert_local_path(out + ".png"), dpi=200, bbox_inches="tight")
    pd.DataFrame(stats).to_csv(assert_local_path(os.path.join(LOGDIR, f"fig1e_scatter_d2raw_stats{sfx}.csv")), index=False)
    pd.concat(persess, ignore_index=True).to_csv(
        assert_local_path(os.path.join(LOGDIR, f"fig1e_scatter_d2raw_per_session{sfx}.csv")), index=False)
    print(f"\nwrote {out}.svg/.png, fig1e_scatter_d2raw_stats{sfx}.csv, fig1e_scatter_d2raw_per_session{sfx}.csv")


if __name__ == "__main__":
    main()
