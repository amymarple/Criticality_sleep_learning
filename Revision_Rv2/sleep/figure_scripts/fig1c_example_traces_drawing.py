"""Drawing functions from analysis_scripts/revision/ (fig1c_example_traces.py, run_fig1_metrics.py) used by the
figure notebooks. OUT: the folder of the notebook's tables, set by the notebook.
"""
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import numpy as np
import os

# matplotlib settings made by importing the scripts
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
                     "axes.unicode_minus": False})

OUT = None


# --- fig1c_example_traces.py
T0, T1 = 5200.0, 10000.0


# --- fig1c_example_traces.py
BIN, CAP_PCT, LONG_S, DS_S, BREAK_PAD = 0.04, 99.5, 2.0, 1.0, 10.0


# --- fig1c_example_traces.py
STATE_COL = {"WAKE": "#fdb863", "NREM": "#7b3294", "REM": "#d7b5d8"}


# --- fig1c_example_traces.py
FIGSIZE, LW, STATE_LABELPAD, DPI = (15, 13), 1.5, 30, 250


# --- fig1c_example_traces.py
D2_LABEL = "d2 (2024\nEuclidean)"


from final_figs import assert_local_path   # writes go to ../figure_output only


# --- fig1c_example_traces.py (a drawing section, as a function)
def draw_figure(D, iv, tb, fr, fr_cap):
    """Draw the Fig 1c state bar, population FR and per-panel trace rows from D, iv, tb, fr, fr_cap."""
    edges = np.unique(np.concatenate([x.ravel() for x in iv.values()]))
    edges = edges[(edges >= T0) & (edges <= T1)]
    ROWS = [("EI", "E/I balance\nlog(E / I rate)"), ("synchrony", "synchrony\n(E units active / 50 ms)"),
            ("theta", "theta power\n(ratio, 0–1)"), ("swr_rate", "SWR rate (Hz)"), ("d2", D2_LABEL)]
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    gs = fig.add_gridspec(2 + len(ROWS), 1, height_ratios=[0.12, 0.6] + [0.5] * len(ROWS))
    ax_bar = fig.add_subplot(gs[0])
    for k in ("WAKE", "NREM", "REM"):
        for a, b in iv[k]:
            a, b = max(a, T0), min(b, T1)
            if b > a:
                ax_bar.axvspan(a, b, color=STATE_COL[k], lw=0)
    ax_bar.set_xlim(T0, T1); ax_bar.set_xticks([]); ax_bar.set_yticks([])
    ax_bar.set_ylabel("State", rotation=0, labelpad=STATE_LABELPAD, va="center")
    for s in ("right", "top", "left"):
        ax_bar.spines[s].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, color=STATE_COL[k]) for k in ("WAKE", "NREM", "REM")]
    ax_bar.legend(h, ["Wake", "NREM", "REM"], ncol=3, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0))

    ax = fig.add_subplot(gs[1], sharex=ax_bar)
    m = (tb >= T0) & (tb <= T1)
    r = np.clip(fr[m], 0, fr_cap)   # fr_cap: np.percentile(fr, CAP_PCT) over the whole session; fr here covers the plotted range only
    sm = gaussian_filter1d(r, sigma=max(LONG_S / BIN, 1))
    step = max(int(round(DS_S / BIN)), 1); nb_ = len(sm) // step
    ax.plot(tb[m][:nb_ * step].reshape(nb_, step).mean(1), sm[:nb_ * step].reshape(nb_, step).mean(1),
            lw=LW, color="#525252")
    ax.set_ylabel("Population FR (Hz)")

    t2 = D.center_s.to_numpy(float); s2 = D.state.to_numpy()
    gap = np.zeros(t2.size, bool)
    for e in edges:
        gap |= (t2 >= e - BREAK_PAD) & (t2 <= e + BREAK_PAD)
    brk = np.r_[True, (s2[1:] != s2[:-1]) | (np.diff(t2) > 5 * np.median(np.diff(t2)))]
    starts = np.flatnonzero(brk); ends = np.r_[starts[1:], t2.size]
    axes = []
    for j, (col, lab) in enumerate(ROWS):
        ax = fig.add_subplot(gs[2 + j], sharex=ax_bar); axes.append(ax)
        y = D[col].to_numpy(float).copy()
        if col == "d2":
            y = np.clip(y, 0, np.nanpercentile(y, 99.5))
        y[gap] = np.nan
        for e in edges:
            ax.axvspan(max(e - BREAK_PAD, T0), min(e + BREAK_PAD, T1), ymin=0.9, ymax=1.0, facecolor="0.8",
                       alpha=0.5, lw=1, zorder=0)
        for a, b in zip(starts, ends):
            if b - a >= 2:
                ax.plot(t2[a:b], y[a:b], lw=LW, color=STATE_COL.get(s2[a], "0.5"), solid_capstyle="round")
        ax.set_ylabel(lab)
    ax = axes[-1]
    ax.xaxis.set_major_locator(MultipleLocator(base=600, offset=T0))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{(x - T0) / 60:.0f}"))
    ax.set_xlim(T0, T1); ax.set_xlabel("Time (min)")
    for a_ in fig.axes[:-1]:
        a_.tick_params(labelbottom=False)
    ax_bar.tick_params(bottom=False)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{OUT}.{ext}"), dpi=DPI, bbox_inches="tight")
