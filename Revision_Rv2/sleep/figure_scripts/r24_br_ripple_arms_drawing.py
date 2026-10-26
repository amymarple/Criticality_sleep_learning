"""Drawing functions from analysis_scripts/revision/ (r24_br_ripple_arms.py, d2_arms_raw.py) used by the
figure notebooks. LOGDIR: the folder of the notebook's tables, set by the notebook.
"""
import numpy as np
import os
import pandas as pd
from scipy import stats

LOGDIR = None


# --- d2_arms_raw.py
GREY, RED, BLUE = "#7f7f7f", "#d62728", "#1f77b4"   # BLUE: excised arm of barrage-rate panels


# --- d2_arms_raw.py
X6 = np.arange(1, 7)


# --- d2_arms_raw.py
def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


# --- d2_arms_raw.py
def fisher(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    z = np.arctanh(np.clip(v, -0.999, 0.999))
    return float(np.tanh(z.mean())), float(np.tanh(z.std(ddof=1) / np.sqrt(len(z)))) if len(z) > 1 else np.nan, len(v)


# --- d2_arms_raw.py
def sextile_curve(W, d2col, ratecol):
    Y = []
    for s, g in W.groupby("session"):
        g = g.dropna(subset=[d2col, ratecol])
        if len(g) < 12:
            continue
        sex = pd.qcut(g[ratecol].rank(method="first"), 6, labels=False) + 1
        m = g.groupby(sex)[d2col].mean()
        Y.append([m.get(k, np.nan) for k in X6])
    Y = np.array(Y)
    return np.nanmean(Y, 0), np.nanstd(Y, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(Y), 0)), len(Y)


# --- r24_br_ripple_arms.py
# POP=all: m on all CA1 units (rate > 0.2 Hz), the population d2 uses. Default: pyramidal.
POP = os.environ.get("POP", "pyr")


# --- r24_br_ripple_arms.py
ARMS = [("d2_full", "d2, full window"), ("d2_excised", "d2, ripple-excised"),
        ("m_full", "m, full window"), ("m_excised", "m, ripple-excised")]


# --- r24_br_ripple_arms.py
D2_YLAB = "d2 (2024 Euclidean, p = 20, 40 ms)"


# --- r24_br_ripple_arms.py
M_YLAB = f"m (MR, {'pyramidal' if POP == 'pyr' else 'all CA1 units'}, 4 ms, k$_{{max}}$ = 200)"


# --- r24_br_ripple_arms.py
def n_dir(t):
    """sessions in the direction of the mean effect"""
    return t["n_pos"] if t["r"] > 0 else t["n"] - t["n_pos"]


# --- r24_br_ripple_arms.py (its tables read from LOGDIR)
def load_table():
    """NREM windows with both d2 arms (d2_arms_raw_windows.csv) joined to both m arms (r24_br_ripple_windows.csv)."""
    W = pd.read_csv(os.path.join(LOGDIR, "d2_arms_raw_windows.csv"), float_precision="round_trip")
    W = W[W.state == "NREM"][["session", "center_s", "d2_full", "d2_excised", "ripple_rate"]]
    M = pd.read_csv(os.path.join(LOGDIR, "r24_br_ripple_windows.csv"), float_precision="round_trip")
    return W.merge(M, on=["session", "center_s"], how="inner")


# --- r24_br_ripple_arms.py
def arm_stats(D, verbose=True):
    """per-session Spearman of each arm with SWR rate; Fisher mean +- SEM, sessions positive, signed-rank"""
    rows = []
    for col, lab in ARMS:
        r = np.array([stats.spearmanr(g[col], g.ripple_rate, nan_policy="omit").statistic for _, g in D.groupby("session")])
        m, se, n = fisher(r)
        rec = dict(arm=col, label=lab, r=m, se=se, n_pos=int((r > 0).sum()), n=n, p=stats.wilcoxon(r).pvalue,
                   per_session=";".join(f"{x:.3f}" for x in r))
        rows.append(rec)
        if verbose:
            print(f"  {lab:20s} {m:+.3f} +- {se:.3f}  {rec['n_pos']}/{n}  p = {rec['p']:.4f}")
    return rows


# --- r24_br_ripple_arms.py
def sextile_panel(ax, D, T, full, exc, ylab, title, legend_loc="lower right", count_y=0.97):
    """raw values by within-session SWR-rate sextile, full (grey) vs ripple-excised (red)"""
    for k, (col, c, lab) in enumerate(((full, GREY, "full window"), (exc, RED, "ripple-excised"))):
        m, e, n = sextile_curve(D, col, "ripple_rate")
        ax.errorbar(X6, m, yerr=e, fmt="o-", capsize=4, lw=2, ms=5, color=c, label=lab, zorder=3 - k)
        t = T[col]
        ax.text(0.03, count_y - 0.09 * k, f"{n_dir(t)}/{t['n']} {'↑' if t['r'] > 0 else '↓'} {stars(t['p'])}",
                transform=ax.transAxes, va="top", color=c, fontsize=9)
    ax.set_xticks(X6); ax.set_xlabel("SWR rate sextile (within session)"); ax.set_ylabel(ylab + "\nraw, mean ± SEM over sessions")
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)
    ax.set_title(title, loc="left", fontweight="bold")
