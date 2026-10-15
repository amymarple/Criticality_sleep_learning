"""
fig1c_example_traces.py -- Fig. 1c (the PPP20_day5 example) with the other window-level measures
stacked under the population rate: E/I balance, synchrony, theta power and SWR rate, and d2 last.

REPRODUCES the published panel (`notebooks/figures/Fig1_C_Example_plot_mouse.ipynb`, left untouched):
its function definitions (bin_spikes, clean_and_reassign_epochs and helpers) and its `corrections`
list are executed from the notebook's own cells, so the state bar, the state corrections (bridge 2 s),
the MUA row (all CA1 units, 40 ms, capped at the 99.5th percentile, Gaussian 2 s, shown at 1 s) and
the d2 row (d2 (2024 Euclidean, p = 20) from *_D2_time_resolved_error_bar_forplot.mat, 30 s windows
every 2 s, coloured by the .mat state, broken and shaded +-10 s around every transition) are the
published ones. Range 5200-10000 s.

NEW ROWS, one value per d2 window (the same 30 s window at the same centre), with the revision
pipeline's definitions (units above MIN_FIRING_RATE, run_fig1_metrics.load_session):
  E/I       log(E rate / I rate), pyramidal vs interneuron spikes over the window (run_fig1_metrics)
  synchrony fraction of E units active per 50 ms, subsampled to 81 E units, median of 20 draws
            (fig1_metrics.active_matrix + synchrony_subsampled_from_matrix, as synchrony_sub81)
  theta     buzcode theta ratio (theta 5-10 Hz above 1/f, 0-1 within session), window mean
            (fig1_theta_ratio.py)
  SWR rate  ripple START times in the window / 30 s (fig1_dropin_metrics)
drawn exactly as the d2 row (state colours, transition gaps).

Reads the share read-only; writes only to D:.
    -> data/revision/log/fig1c_example_traces.csv
    -> reports/revision_figs/fig1c_example_traces.png / .svg
"""
from __future__ import annotations

import ast
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MultipleLocator
from scipy.ndimage import gaussian_filter1d

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path, load_session                      # noqa: E402
from fig1_metrics import (WINDOW_S, active_matrix, load_d2_timeline, slice_sorted,  # noqa: E402
                          synchrony_subsampled_from_matrix)
from fig1_dropin_metrics import SYNC_TARGET_FIXED                                  # noqa: E402
from fig1_theta_ratio import load_theta_ratio, theta_ratio_window                  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
NB = os.path.join(REPO, "notebooks", "figures", "Fig1_C_Example_plot_mouse.ipynb")
BP, REGION = r"W:\data\PPP\PPP20\day5", "CA1"
T0, T1 = 5200.0, 10000.0                          # the notebook's call
BIN, CAP_PCT, LONG_S, DS_S, BREAK_PAD = 0.04, 99.5, 2.0, 1.0, 10.0
OUT_CSV = os.path.join(REPO, "data", "revision", "log", "fig1c_example_traces.csv")
OUT = os.path.join(REPO, "reports", "revision_figs", "fig1c_example_traces")
STATE_COL = {"WAKE": "#fdb863", "NREM": "#7b3294", "REM": "#d7b5d8"}
# figure geometry; a final-size figure overrides these (notebooks/figures_revision/Fig1_AB_hypothesis_example.ipynb)
FIGSIZE, LW, STATE_LABELPAD, DPI = (15, 13), 1.5, 30, 250
D2_LABEL = "d2 (2024\nEuclidean)"

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 12, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none",
                     "axes.unicode_minus": False})


def notebook_defs():
    """Function definitions and the `corrections` list from the notebook's cells 5 and 13."""
    nb = json.load(open(NB, encoding="utf-8"))
    ns = {"np": np, "nel": __import__("nelpy")}
    for i in (5, 13):
        tree = ast.parse("".join(nb["cells"][i]["source"]))
        keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))
                or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "corrections" for t in n.targets))]
        exec(compile(ast.Module(body=keep, type_ignores=[]), f"cell{i}", "exec"), ns)
    return ns


def main():
    import nelpy as nel
    from neuro_py.io import loading
    ns = notebook_defs()
    # --- the published rows -------------------------------------------------------------------
    st, cm = loading.load_spikes(BP, brainRegion=REGION)
    sd = loading.load_SleepState_states(BP)
    nrem, wake, rem = ns["clean_and_reassign_epochs"](
        nel.EpochArray(sd["NREMstate"]), nel.EpochArray(sd["WAKEstate"]), nel.EpochArray(sd["REMstate"]),
        ns["corrections"], bridge_gap_sec=2.0)
    tb, fr = ns["bin_spikes"](st.time, bin_size=BIN)
    D2, src = load_d2_timeline(BP, REGION)
    print(f"d2 source: {src}")
    assert "forplot" in src, "not the published d2 timeline"
    D2 = D2[(D2.center_s >= T0) & (D2.center_s <= T1)].sort_values("center_s").reset_index(drop=True)
    stcol = "d2_state" if "d2_state" in D2.columns else "state"
    D2["state"] = D2[stcol].astype(str).str.upper()

    # --- the new rows, same windows as d2 --------------------------------------------------------
    unit_spikes, is_e, is_i, _ = load_session(BP, REGION)
    e_sp = [unit_spikes[i] for i in np.flatnonzero(is_e)]
    e_flat = np.sort(np.concatenate(e_sp)); i_flat = np.sort(np.concatenate([unit_spikes[i] for i in np.flatnonzero(is_i)]))
    th_t, th = load_theta_ratio(BP)
    rip = np.sort(np.asarray(loading.load_ripples_events(basepath=BP)["start"], float))
    rng = np.random.default_rng(0)
    rows = []
    for c in D2.center_s.to_numpy(float):
        a, b = c - WINDOW_S / 2, c + WINDOW_S / 2
        ne, ni = slice_sorted(e_flat, a, b).size, slice_sorted(i_flat, a, b).size
        ei = float(np.log((ne / WINDOW_S + 1e-12) / (ni / WINDOW_S + 1e-12))) if ni > 0 else np.nan
        syn, _ = synchrony_subsampled_from_matrix(active_matrix(e_sp, a, b), SYNC_TARGET_FIXED, rng)
        tv, _ = theta_ratio_window(th_t, th, a, b)
        swr = (np.searchsorted(rip, b, "right") - np.searchsorted(rip, a, "left")) / WINDOW_S
        rows.append(dict(EI=ei, synchrony=syn, theta=tv, swr_rate=swr))
    D = pd.concat([D2[["center_s", "state", "d2"]], pd.DataFrame(rows)], axis=1)
    D.to_csv(assert_local_path(OUT_CSV), index=False)
    print(f"{len(D)} windows in [{T0:.0f}, {T1:.0f}] s; {int(is_e.sum())} E / {int(is_i.sum())} I units above threshold")
    print(D.groupby("state")[["EI", "synchrony", "theta", "swr_rate", "d2"]].median().round(3).to_string())

    # --- figure -------------------------------------------------------------------------------------
    iv = {k: np.asarray(e.data, float).reshape(-1, 2) for k, e in (("WAKE", wake), ("NREM", nrem), ("REM", rem))}
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

    ax = fig.add_subplot(gs[1], sharex=ax_bar)                     # MUA exactly as the notebook
    m = (tb >= T0) & (tb <= T1)
    r = np.clip(fr[m], 0, np.percentile(fr, CAP_PCT))
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
            y = np.clip(y, 0, np.nanpercentile(y, 99.5))            # the notebook's d2 cap
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
    print(f"-> {OUT}.png/.svg")


if __name__ == "__main__":
    main()
