"""Drawing functions from analysis_scripts/revision/ (r25_adf_box.py) used by the figure notebooks. LOGDIR,
FIGDIR: the folder of the notebook's tables, set by the notebook. main() takes the per-session table that
r25_adf_box.per_session() derives from r25_stationarity_windows.csv.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from final_figs import assert_local_path   # the producers' write guard; the figures are captured, not written

LOGDIR = None
FIGDIR = None


# --- r25_adf_box.py
L = 30


# --- r25_adf_box.py
MIN_WIN = 5


# --- r25_adf_box.py
ALPHA = 0.05


# --- r25_adf_box.py
STATES = ["NREM", "REM", "THETA", "nonTHETA"]


# --- r25_adf_box.py
STATE_LABEL = {"NREM": "NREM", "REM": "REM", "THETA": "Wake θ", "nonTHETA": "Wake non-θ"}


# --- r25_adf_box.py
STATE_COLOR = {"THETA": "#d95f0e", "nonTHETA": "#fdb863", "NREM": "#7b3294", "REM": "#d7b5d8"}


# --- r25_adf_box.py
VARIANTS = [("real", "data", 1.0), ("stationary", "stationary AR(20) surrogate", 0.5),
            ("rw", "data + random walk (1 SD)", 0.18)]


# --- r25_adf_box.py
def _wilcoxon(d):
    d = np.asarray(d, float)
    if d.size < 3 or np.all(d == 0):
        return 1.0
    return float(stats.wilcoxon(d).pvalue)


# --- r25_adf_box.py
def paired_stats(F):
    rows = []
    for st in STATES:
        P = F[(F.state == st) & F.included].pivot(index="sess", columns="variant", values="frac_reject")
        for b in ("stationary", "rw"):
            if "real" not in P or b not in P:
                continue
            d = (P["real"] - P[b]).dropna()
            rows.append(dict(state=st, contrast=f"real - {b}", n_sessions=len(d),
                             median_real=float(P.loc[d.index, "real"].median()),
                             median_other=float(P.loc[d.index, b].median()),
                             median_diff=float(d.median()), n_real_higher=int((d > 0).sum()),
                             n_real_lower=int((d < 0).sum()), p=_wilcoxon(d)))
    return pd.DataFrame(rows)


# --- r25_adf_box.py
def _p_text(p):
    if p >= 0.05:
        return f"n.s. (p = {p:.2f})"
    return "***" if p < .001 else "**" if p < .01 else "*"


# --- r25_adf_box.py
def plot(F, T):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 10, "axes.linewidth": 1.2})
    rng = np.random.default_rng(0)
    off = np.array([-0.27, 0.0, 0.27])
    width = 0.22
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for i, st in enumerate(STATES):
        G = F[(F.state == st) & F.included]
        P = G.pivot(index="sess", columns="variant", values="frac_reject")
        jit = pd.Series(rng.uniform(-0.05, 0.05, len(P)), index=P.index)
        for (v, _, alpha), dx in zip(VARIANTS, off):
            y = P[v].dropna() if v in P else pd.Series(dtype=float)
            if not len(y):
                continue
            ax.boxplot([y.to_numpy()], positions=[i + dx], widths=width, patch_artist=True,
                       showfliers=False, whis=(0, 100),
                       boxprops=dict(facecolor=mpl.colors.to_rgba(STATE_COLOR[st], alpha),
                                     edgecolor="k", lw=1.0),
                       medianprops=dict(color="k", lw=1.6),
                       whiskerprops=dict(color="k", lw=1.0), capprops=dict(color="k", lw=1.0))
            ax.scatter(i + dx + jit[y.index], y, s=9, color="k", alpha=0.6, lw=0, zorder=3)
        if "stationary" in P:
            for s, r in P[["real", "stationary"]].dropna().iterrows():
                ax.plot([i + off[0] + jit[s], i + off[1] + jit[s]], [r.real, r.stationary],
                        color="0.6", lw=0.6, zorder=2)
        for b, x1, y0 in (("stationary", off[1], 1.04), ("rw", off[2], 1.13)):
            t = T[(T.state == st) & (T.contrast == f"real - {b}")]
            if not len(t):
                continue
            t = t.iloc[0]
            ax.plot([i + off[0], i + off[0], i + x1, i + x1], [y0 - .015, y0, y0, y0 - .015], color="k", lw=0.8)
            lab = _p_text(t.p) if b == "stationary" else f"{t.n_real_higher}/{t.n_sessions} {_p_text(t.p)}"
            ax.text(i + (off[0] + x1) / 2, y0 + .005, lab, ha="center", va="bottom", fontsize=7.5)
    n = {st: int(((F.state == st) & (F.variant == "real") & F.included).sum()) for st in STATES}
    ax.set_xticks(range(len(STATES)))
    ax.set_xticklabels([f"{STATE_LABEL[s]}\n(n = {n[s]})" for s in STATES])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_xlim(-0.6, len(STATES) - 0.4)
    ax.set_ylim(-0.03, 1.22)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel(f"fraction of {L} s windows rejecting a unit root\n(ADF, p < {ALPHA:g})")
    ax.legend(handles=[Patch(facecolor=mpl.colors.to_rgba("0.3", a), edgecolor="k", label=lab)
                       for _, lab, a in VARIANTS],
              frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    assert_local_path(FIGDIR)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"r25_adf_box.{ext}")), dpi=300, bbox_inches="tight")
    print(f"figure -> {FIGDIR}\\r25_adf_box.png/.svg")


# --- r25_adf_box.py (main, with the per-session table F as its input)
def main(F):
    """Paired statistics of F (session x state x variant: fraction of 30 s windows with ADF p < 0.05), printed,
    then the figure."""
    T = paired_stats(F)
    for st in STATES:
        g = F[(F.state == st) & (F.variant == "real")]
        print(f"{st:9s} sessions {len(g)}, with >= {MIN_WIN} windows {int(g.included.sum())} "
              f"(windows/session {g.n_windows.min()}-{g.n_windows.max()})")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(T.round(4).to_string(index=False))
    plot(F, T)
