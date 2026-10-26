"""Drawing functions from analysis_scripts/revision/ (letter_figs_r12_r13.py, run_fig1_metrics.py) used by the
figure notebooks. FIGDIR: the folder its figure is saved to (the notebook captures the figure instead).
"""
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
FIGDIR = "."


# --- letter_figs_r12_r13.py
STATES = ["NREM", "nonTHETA", "REM", "THETA"]


# --- letter_figs_r12_r13.py
LAB = {"NREM": "NREM", "nonTHETA": "non-theta wake", "REM": "REM", "THETA": "theta wake"}


# --- letter_figs_r12_r13.py
COL = {"NREM": "#7b3294", "REM": "#d7b5d8", "THETA": "#d95f0e", "nonTHETA": "#fdb863"}   # Fig-1 state colours


# --- letter_figs_r12_r13.py
ORDERS = [5, 10, 15, 20, 30, 40]


# --- letter_figs_r12_r13.py (when it is imported)
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


from final_figs import assert_local_path   # writes go to ../figure_output only


# --- letter_figs_r12_r13.py
def save(fig, name):
    out = os.path.join(FIGDIR, name)
    for _ in range(4):
        try:
            fig.savefig(assert_local_path(out), dpi=200, bbox_inches="tight"); fig.savefig(out[:-4] + ".svg", bbox_inches="tight"); break
        except OSError:
            import time; time.sleep(3)
    print("  ->", out)


# --- letter_figs_r12_r13.py (r12, with its table LB as the argument)
def draw_r12(LB):
    """AR order vs residual whiteness: the fraction of windows whose AR(p) residuals pass the Ljung-Box test
    (h = 50 lags, df = h - p), by state and order, from the window x order table LB (ljungbox_df_audit.csv)."""
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    # y axis just "Fraction of windows with white residuals" (the test, the peaks and the session / window counts
    # go in the legend, not on the figure)
    W = LB.pivot_table(index="state", columns="order", values="white_df", aggfunc="mean")
    for s in STATES:
        ax.plot(W.columns, W.loc[s], "o-", color=COL[s], lw=2, ms=4, label=LAB[s])
    ax.axvline(20, color="#bbbbbb", lw=6, alpha=0.4, zorder=0)
    ax.set_xlabel("AR order p (40 ms bins)"); ax.set_ylabel("Fraction of windows with white residuals")
    ax.set_xticks(ORDERS)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); ax.set_ylim(0.6, 1.0)
    print(f"whiteness (df-corrected), %: 18 sessions, {LB[LB.order == 20].shape[0]} windows "
          f"({LB[LB.order == 20].groupby('state').size().to_dict()})"); print((100 * W).round(1).to_string())
    fig.tight_layout(); save(fig, "r12_order_selection.png")
