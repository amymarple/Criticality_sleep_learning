"""Drawing functions from analysis_scripts/revision/ (r25_drift_control_rates.py) used by the figure notebooks.
LOGDIR, FIGDIR: the folder of the notebook's tables, set by the notebook (window_length_panel reads
window_length_sweep_windows.csv from LOGDIR).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from final_figs import assert_local_path   # the producers' write guard; the figures are captured, not written

LOGDIR = None
FIGDIR = None


# --- r25_drift_control_rates.py
ARMS = ["real", "detrended", "partial", "low-drift"]


# --- r25_drift_control_rates.py
def fisher(v):
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)])
    if v.size == 0:
        return np.nan, np.nan, 0
    z = np.arctanh(np.clip(v, -0.999999, 0.999999))
    return float(np.tanh(z.mean())), float(z.std(ddof=1) / np.sqrt(z.size)) if z.size > 1 else np.nan, z.size


# --- r25_drift_control_rates.py
RELS = [("fig1", "ripple_rate", "Fig 2 cohort: d2 ~ ripple rate", +1),
        ("barrage", "ripple_rate", "Fig 3 cohort: d2 ~ ripple rate", +1),
        ("barrage", "barrage_rate", "Fig 3 cohort: d2 ~ barrage rate", -1)]


# --- r25_drift_control_rates.py
def report(R):
    print("\n" + "=" * 112)
    print("d2 (2024 Euclidean, p=20, 40 ms) ~ event rate, NREM 30 s non-overlapping windows; session = unit")
    print("arms: real | linearly detrended d2 | partial r given |trend| | windows with |trend| < 0.5 SD only")
    print("=" * 112)
    for cohort, rate_col, label, sign in RELS:
        P_ = R[(R.cohort == cohort) & (R.rate_col == rate_col)]
        if not len(P_):
            continue
        print(f"\n  {label}  (n = {P_.session.nunique()} sessions)")
        print(f"  {'arm':12s}{'Fisher r':>10s}{'± SE':>8s}{'sign-consistent':>17s}{'signed-rank p':>15s}")
        for arm in ARMS + ["trend_vs_rate", "trend_vs_d2"]:
            v = P_[arm].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size < 3:
                continue
            fr, se, n = fisher(v)
            cons = int((np.sign(v) == sign).sum()) if arm in ARMS else int((v < 0).sum())
            lab = arm if arm in ARMS else ("|trend| ~ rate" if arm == "trend_vs_rate" else "|trend| ~ d2")
            note = "" if arm in ARMS else ("  (neg count)")
            print(f"  {lab:12s}{fr:+10.3f}{se:8.3f}{cons:>9d}/{n:<7d}{stats.wilcoxon(v).pvalue:15.4f}{note}")


# --- r25_drift_control_rates.py
def window_length_panel(ax, rng, title):
    """d: the ripple-rate relation at shorter (and longer) windows, computed on the window
    table in window_length_sweep_windows.csv (NREM, non-overlapping, d2 (2024 Euclidean,
    p = 20, 40 ms) by Yule-Walker, all CA1 units), per-session Spearman as panels a-c."""
    W = pd.read_csv(os.path.join(LOGDIR, "window_length_sweep_windows.csv"))
    W = W[W.state == "NREM"].dropna(subset=["d2ex", "ripple_rate"])
    lengths = sorted(W.window_s.unique())
    for i, L in enumerate(lengths):
        v = np.array([stats.spearmanr(h.d2ex, h.ripple_rate).statistic
                      for _, h in W[W.window_s == L].groupby(["animal", "day"])
                      if len(h) >= 10 and h.ripple_rate.nunique() > 2])
        ax.plot(i + rng.normal(0, 0.05, v.size), v, "o", ms=4, color="#7b3294", mec="k", mew=0.4, alpha=0.85)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=2.5)
        ax.text(i, 0.62, f"{int((v > 0).sum())}/{v.size}\np={stats.wilcoxon(v).pvalue:.3f}", ha="center", fontsize=7.5)
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.axvspan(-0.5, 1.5, color="0.92", zorder=0)
    ax.text(0.5, -0.55, "N/p < 20", ha="center", fontsize=7.5, color="0.4")
    ax.set_xticks(range(len(lengths))); ax.set_xticklabels([f"{int(L)} s" for L in lengths], fontsize=8)
    ax.set_xlim(-0.5, len(lengths) - 0.5)
    ax.set_xlabel("window length (non-overlapping)", fontsize=8.5)
    ax.set_title(title, fontsize=9.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# --- r25_drift_control_rates.py
def drift_panel(ax, R, cohort, rate_col, title, sign, rng, col):
    P_ = R[(R.cohort == cohort) & (R.rate_col == rate_col)]
    for i, arm in enumerate(ARMS):
        v = P_[arm].to_numpy(float)
        v = v[np.isfinite(v)]
        ax.plot(i + rng.normal(0, 0.05, v.size), v, "o", ms=4, color=col[arm], mec="k", mew=0.4, alpha=0.85)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=2.5)
        pw = stats.wilcoxon(v).pvalue if v.size > 2 else np.nan
        ax.text(i, 0.62, f"{int((np.sign(v) == sign).sum())}/{v.size}\np={pw:.3f}", ha="center", fontsize=7.5)
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(["d2 as\npublished", "d2\ndetrended", "partial r\n(| trend)", "low-drift\nwindows"], fontsize=8)
    ax.set_title(title, fontsize=9.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# --- r25_drift_control_rates.py
def plot(R, full=False):
    """a = SWR-rate relation by window length, b = SWR-rate relation under drift control
    (Fig-1 cohort), c = barrage-rate relation under drift control (barrage cohort); the
    barrage cohort's SWR-rate panel is dropped. full=True redraws the four-panel
    version (a-c drift control incl. that panel, d window length) to backup/."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 10, "axes.linewidth": 1.2})
    col = {"real": "#7b3294", "detrended": "#9970ab", "partial": "#c2a5cf", "low-drift": "#e7d4e8"}
    rng = np.random.default_rng(0)
    if full:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
        for ax, (cohort, rate_col, label, sign), letter in zip(axes, RELS, "abc"):
            drift_panel(ax, R, cohort, rate_col, f"{letter}  {label}", sign, rng, col)
        window_length_panel(axes[3], rng, "d  Fig 2 cohort: d2 ~ ripple rate by window length")
        out_dir, name = os.path.join(FIGDIR, "backup"), "r25_drift_control_rates_4panel"
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), sharey=True, gridspec_kw={"width_ratios": [1.15, 1, 1]})
        window_length_panel(axes[0], rng, "a  d2 ~ SWR rate by window length")
        drift_panel(axes[1], R, "fig1", "ripple_rate", "b  d2 ~ SWR rate, drift controlled", +1, rng, col)
        drift_panel(axes[2], R, "barrage", "barrage_rate", "c  d2 ~ barrage rate, drift controlled", -1, rng, col)
        out_dir, name = FIGDIR, "r25_drift_control_rates"
    axes[0].set_ylabel("per-session Spearman r")
    axes[0].set_ylim(-0.6, 0.75)
    fig.tight_layout()
    os.makedirs(assert_local_path(out_dir), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(out_dir, f"{name}.{ext}")), dpi=300)
    print(f"\nfigure -> {os.path.join(out_dir, name)}.png/.svg")
