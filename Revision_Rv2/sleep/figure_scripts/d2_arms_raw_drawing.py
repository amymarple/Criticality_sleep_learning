"""Drawing functions from analysis_scripts/revision/ (d2_arms_raw.py, run_fig1_metrics.py) used by the figure
notebooks. FIGDIR: the folder its figure is saved to (the notebook captures the figure instead).
"""
import os
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
FIGDIR = "."


# --- d2_arms_raw.py
# LABEL=centre: a wake window is THETA if its centre lies in SleepState THETA (the split of Fig 1), on the
# non-overlapping _error_bar.mat windows of r24_state_centre.py; LABEL=wholewindow: the whole window in the state.
LABEL = os.environ.get("LABEL", "centre")


# --- d2_arms_raw.py
STATES = ["NREM", "REM", "THETA", "nonTHETA"]


# --- d2_arms_raw.py
STATE_LABEL = {"NREM": "NREM", "REM": "REM", "THETA": "theta wake", "nonTHETA": "non-theta wake"}


# --- d2_arms_raw.py
XPOS = np.array([0, 1, 2.5, 3.5])


# --- d2_arms_raw.py
PAIRS = [("NREM", "REM"), ("nonTHETA", "THETA")]


# --- d2_arms_raw.py
GREY, RED, BLUE = "#7f7f7f", "#d62728", "#1f77b4"   # BLUE: the events-excised arm of the barrage-rate panel


# --- d2_arms_raw.py
def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


from final_figs import assert_local_path   # writes go to ../figure_output only


# --- d2_arms_raw.py (a drawing section, as a function: main()'s states block and its panel a)
def draw_main(Fs):
    """Panel a of d2_arms_raw.main(): per-session median d2 (2024 Euclidean, p = 20) by state, full window and
    ripple-excised, mean +- SEM over sessions, with the within-sleep (NREM vs REM) and within-wake (non-theta vs
    theta) paired Wilcoxon tests, from the window table Fs (both arms per window). main()'s panels b-d (SWR,
    barrage and replay rate) are not part of this figure and are left out; their axes stay empty."""
    rows = []

    # -- states: per-session median per state, both arms
    S = Fs.groupby(["session", "state"])[["d2_full", "d2_excised"]].median().reset_index()
    print("\nSTATES -- per-session median d2 (2024 Euclidean, p = 20), mean +- SEM over sessions; paired vs NREM")
    print(f"  {'state':16s}{'n':>4s} | {'full':>8s} {'+-':>7s} | {'excised':>8s} {'+-':>7s} | vs NREM full (n<NREM /n, p) | excised")
    for stt in STATES:
        g = S[S.state == stt]
        line = f"  {STATE_LABEL[stt]:16s}{len(g):4d} |"
        for arm in ("d2_full", "d2_excised"):
            line += f" {g[arm].mean():8.4f} {g[arm].sem():7.4f} |"
            rows.append(dict(block="state", arm=arm, state=stt, mean=g[arm].mean(), sem=g[arm].sem(), n=len(g)))
        if stt != "NREM":
            for arm in ("d2_full", "d2_excised"):
                P = S[S.state == "NREM"][["session", arm]].merge(g[["session", arm]], on="session", suffixes=("_nrem", "_x")).dropna()
                d = P[f"{arm}_nrem"] - P[f"{arm}_x"]
                p = stats.wilcoxon(d).pvalue if len(d) > 2 else np.nan
                line += f" {int((d > 0).sum())}/{len(d)} p={p:.4f} |"
                rows.append(dict(block="state_contrast", arm=arm, state=stt, delta=d.median(), n_pos=int((d > 0).sum()), n=len(d), p=p))
        print(line)
    # within sleep and within wake only (no sleep-vs-wake contrasts in the figures)
    print("  within-pair contrasts (paired, per-session medians):")
    for a, b in PAIRS:
        for arm in ("d2_full", "d2_excised"):
            P = S[S.state == a][["session", arm]].merge(S[S.state == b][["session", arm]], on="session", suffixes=("_a", "_b")).dropna()
            d = P[f"{arm}_a"] - P[f"{arm}_b"]
            p = stats.wilcoxon(d).pvalue if len(d) > 2 else np.nan
            print(f"    {a} - {b:9s} {arm:11s} median diff {d.median():+.4f}  {int((d > 0).sum())}/{len(d)} {a} higher  p = {p:.4f}")
            rows.append(dict(block="pair_contrast", arm=arm, state=f"{a}-{b}", delta=d.median(), n_pos=int((d > 0).sum()), n=len(d), p=p))

    # -- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.5))
    ax = axes[0]
    for k, (arm, c, lab) in enumerate((("d2_full", GREY, "full window"), ("d2_excised", RED, "ripple-excised"))):
        x = XPOS + (k - 0.5) * 0.18
        m = [S[S.state == s][arm].mean() for s in STATES]; e = [S[S.state == s][arm].sem() for s in STATES]
        ax.errorbar(x, m, yerr=e, fmt="o", capsize=4, ms=5, color=c, label=lab, zorder=3)
        for s, g in S.groupby("session"):
            g = g.set_index("state").reindex(STATES)[arm].to_numpy()
            for i in (0, 2):                                            # lines within sleep / within wake only
                ax.plot(x[i:i + 2], g[i:i + 2], "-", color=c, alpha=0.18, lw=0.8)
    ymax = ax.get_ylim()[1]
    for (a, b), x0 in zip(PAIRS, (XPOS[0], XPOS[2])):
        for k, (arm, c) in enumerate((("d2_full", GREY), ("d2_excised", RED))):
            rec = [r for r in rows if r["block"] == "pair_contrast" and r["arm"] == arm and r["state"] == f"{a}-{b}"][0]
            ax.text(x0 + 0.5, ymax * (1.0 - 0.07 * k), f"{rec['n_pos']}/{rec['n']} {stars(rec['p'])}",
                    ha="center", va="bottom", color=c, fontsize=8)
    ax.set_ylim(top=ymax * 1.1)
    ax.set_xticks(XPOS); ax.set_xticklabels([STATE_LABEL[s] for s in STATES], rotation=20, ha="right")
    ax.set_ylabel("d2 (2024 Euclidean, p = 20), raw\nsession median, mean ± SEM")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title(f"a  states ({'wake split by window centre' if LABEL == 'centre' else 'whole window in state'}), "
                 f"{S.session.nunique()} sessions", loc="left", fontweight="bold", fontsize=8.5)
    fig.suptitle("Both Euclidean arms in raw units -- grey = full window (published), red = ripple-excised, blue = ripples + barrages excised; "
                 "non-overlapping 30 s windows, session as the unit", fontsize=9, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGDIR, "d2_arms_raw.png") if LABEL == "centre" else \
        os.path.join(FIGDIR, "backup", f"d2_arms_raw_{LABEL}.png")
    for i in range(4):
        try:
            fig.savefig(assert_local_path(out), dpi=200, bbox_inches="tight"); fig.savefig(out[:-4] + ".svg", bbox_inches="tight"); break
        except OSError:
            import time; time.sleep(3)
    print("  ->", out)
