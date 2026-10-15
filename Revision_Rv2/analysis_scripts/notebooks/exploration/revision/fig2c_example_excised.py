"""fig2c_example_excised.py -- Figure 2c example trace: the submitted panel plus the ripple-excised d2.

Reruns the submitted example-trace notebook as-is, every cell except the final plotting cell, to
get the same window_df: d2 (2024 Euclidean, AR order 20), MUA smoothed over 30 s, and replay
probability (share of SWRs in a centred 30 s window with a replay event within +-80 ms of the
ripple peak). The ripple-excised d2 for the same window comes from the published
day8_CA1_D2_time_resolved_rippleablate.mat (same 30 s / 2 s grid, NREM only, read via
fig1_d2_vs_mua_2x2.nrem_mat), joined on window centre; the full-window .mat read the same way is
checked against the notebook's own d2. Each line is z-scored within the plotted window; x axis in
seconds (3 min window). Example window: PPP8 day8, NREM epoch 16, 180 s from 14350.6 s.

Inputs: Previous_Rv1/figure_scripts/Fig2_C_Example_replayprobalit.ipynb and its CSV inputs;
day8_CA1_D2_time_resolved.mat and its _rippleablate variant.

    -> reports/revision_figs/fig2c_example_excised.png / .svg
    -> data/revision/log/fig2c_example_excised.csv   (the plotted window)
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                       # noqa: E402
from fig1_d2_vs_mua_2x2 import nrem_mat                              # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
NB_DIR = r"D:\Documents\GitHub\Criticality_sleep_learning\Previous_Rv1\figure_scripts"
NB = "Fig2_C_Example_replayprobalit.ipynb"
BASEPATH = r"W:\data\PPP\PPP8\day8"


def run_notebook_until_plot():
    """Execute the notebook's code cells except the last (the plot) in one namespace."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.show = lambda *a, **k: None
    cells = [c for c in json.load(open(os.path.join(NB_DIR, NB), encoding="utf-8"))["cells"]
             if c["cell_type"] == "code"]
    ns = {"display": lambda *a, **k: None, "__name__": "__nb__"}
    cwd = os.getcwd()
    os.chdir(NB_DIR)                                                  # the notebook reads ../csv/
    try:
        np.random.seed(0)                                             # exploration cells draw; the plotted window is fixed
        for c in cells[:-1]:
            src = "\n".join(l for l in "".join(c["source"]).splitlines()
                            if not l.lstrip().startswith(("%", "!")))
            exec(compile(src, NB, "exec"), ns)
    finally:
        os.chdir(cwd)
        plt.close("all")
    return ns


def main():
    ns = run_notebook_until_plot()
    W = ns["window_df"].copy()
    w0 = ns["window_start"]
    print(f"window {w0:.1f}-{w0 + 180:.1f} s, {len(W)} d2 windows")

    full = nrem_mat(BASEPATH, "")
    exc = nrem_mat(BASEPATH, "_rippleablate")
    key = lambda s: s.round(3)
    W["k"] = key(W.center_s)
    chk = W.merge(full.assign(k=key(full.center_s))[["k", "v"]], on="k", how="left")
    assert chk.v.notna().all() and np.allclose(chk.v, chk.d2), "full-window .mat does not match the notebook's d2"
    W = W.merge(exc.assign(k=key(exc.center_s))[["k", "v"]].rename(columns={"v": "d2_excised"}), on="k", how="left")
    assert W.d2_excised.notna().all(), "ripple-excised d2 missing for some plotted windows"

    z = lambda v: (v - v.mean()) / v.std()
    W["d2_z"], W["d2_excised_z"], W["MUA_z"] = z(W.d2), z(W.d2_excised), z(W.MUA)
    W["time_seconds"] = W.center_s - w0
    W.drop(columns="k").to_csv(assert_local_path(os.path.join(LOGDIR, "fig2c_example_excised.csv")), index=False)
    r = W[["d2", "d2_excised", "MUA", "replay_prob"]].corr(method="spearman")
    print(f"within this window: rho(d2 full, excised) {r.loc['d2', 'd2_excised']:+.2f}; "
          f"rho(replay prob, d2 full) {r.loc['replay_prob', 'd2']:+.2f}, (replay prob, d2 excised) {r.loc['replay_prob', 'd2_excised']:+.2f}, "
          f"(replay prob, MUA) {r.loc['replay_prob', 'MUA']:+.2f}  [one 3-min example; descriptive only]")
    print(f"raw d2: full {W.d2.mean():.4f}, excised {W.d2_excised.mean():.4f} (window means)")

    # ---- figure: the original panel's style, plus the excised line; axis in seconds
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "text.usetex": False, "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
                         "font.size": 14, "axes.titlesize": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "legend.fontsize": 13, "axes.unicode_minus": False,
                         "axes.linewidth": 1.5, "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    # Fig 2 palette (fig2_composite: grey full window, red ripple-excised, dark replay); MUA not drawn
    from fig2_composite import GREY, RED, DARK
    c_d2, c_exc, c_rep = GREY, RED, DARK
    fig, ax1 = plt.subplots(figsize=(12, 5))
    l1, = ax1.plot(W.time_seconds, W.d2_z, lw=2, color=c_d2)
    l1b, = ax1.plot(W.time_seconds, W.d2_excised_z, lw=2, color=c_exc)
    ax1.set_ylabel("d2 (z-score within window)")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    ax2 = ax1.twinx()
    l3, = ax2.plot(W.time_seconds, W.replay_prob, lw=2, color=c_rep, ls="--")
    ax2.set_ylabel("Replay probability"); ax2.spines["top"].set_visible(False)
    ax1.set_xlabel("Time (s)")
    ax1.set_xticks([0, 30, 60, 90, 120, 150, 180]); ax1.set_xlim(0, 180)
    ax1.legend([l1, l1b, l3], ["d2, full window (z)", "d2, ripples excised (z)", "Replay probability"],
               loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False)   # above the axes, off the data
    plt.tight_layout()
    out = os.path.join(FIGDIR, "fig2c_example_excised.png")
    fig.savefig(assert_local_path(out), dpi=300); fig.savefig(out[:-4] + ".svg", format="svg")
    print("->", out)


if __name__ == "__main__":
    main()
