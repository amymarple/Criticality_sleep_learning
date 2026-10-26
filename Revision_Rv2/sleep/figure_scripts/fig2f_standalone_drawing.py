"""Drawing functions from analysis_scripts/revision/ (fig2f_standalone.py, r35_replay_ablation_lrt.py, r35_replay_lmm.py) used by the
figure notebooks.
"""
import numpy as np




# --- r35_replay_ablation_lrt.py
SETS = {
    "background":   [("py_excised", "d2"), ("MUA_bg", "MUA (bg)"), ("FR_std_bg_res", "FR std | MUA (bg)"),
                     ("EI_bg", "E/I (bg)")],
    "in-ripple":    [("py_excised", "d2"), ("MUA", "MUA"), ("FR_std_res", "FR std | MUA"),
                     ("EI_log", "E/I")],
}


# --- r35_replay_lmm.py
PRED = SETS["background"]


# --- r35_replay_lmm.py
X = [f"{c}_z" for c, _ in PRED]


# --- r35_replay_lmm.py
LAB = dict(zip(X, [l for _, l in PRED]))


# --- r35_replay_lmm.py
def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


# --- r35_replay_ablation_lrt.py
COLOR = {"py_excised": "#7b3294", "MUA": "#fdb863", "MUA_bg": "#fdb863",
         "CV": "#1b7837", "CV_bg": "#1b7837",
         "FR_std": "#2c7fb8", "FR_std_bg": "#2c7fb8", "FR_std_res": "#2c7fb8",
         "FR_std_bg_res": "#2c7fb8", "MUA_std": "#d95f0e",
         "MUA_std_bg": "#d95f0e", "EI_bg": "#1b7837", "EI_log": "#1b7837"}


# --- fig2f_standalone.py
def draw(ax, D, compact=False):
    """The panel on a given axes. compact=True: the composite's smaller fonts and no
    footnote (the composite's caption carries the conventions)."""
    pooled, per, raw, r, con, ph = (D[k] for k in ("pooled", "per", "raw", "r", "con", "ph"))
    K = len(raw)
    cols = [COLOR[c] for c in raw]
    rng = np.random.default_rng(0)
    fs_lab, fs_br, fs_tick = (8.5, 7.5, 9) if compact else (10, 9, None)
    v = pooled.loc[raw, "drop_one"].to_numpy(float)
    lo, hi = pooled.loc[raw, "bs_lo"].to_numpy(float), pooled.loc[raw, "bs_hi"].to_numpy(float)
    ax.bar(range(K), v, width=0.6, color=cols, edgecolor="k", linewidth=1.5 if compact else 1.8, zorder=2)
    ax.errorbar(range(K), v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", elinewidth=1.5 if compact else 1.8,
                capsize=4 if compact else 5, capthick=1.5 if compact else 1.8, zorder=4)
    top = max(hi.max(), np.percentile(per.to_numpy(float), 98))
    for j, (c, k) in enumerate(zip(raw, X)):
        pv = per[c].to_numpy(float)
        xs = j + rng.uniform(-0.13, 0.13, pv.size)
        inside = pv <= top
        ax.plot(xs[inside], pv[inside], "o", ms=3.5 if compact else 4.5, color="0.25", alpha=0.75, mew=0, zorder=3)
        for x_, y_, s_ in zip(xs[~inside], pv[~inside], per.index[~inside]):
            ax.plot(x_, top, "^", ms=5 if compact else 6, mfc="white", mec="0.25", mew=1.2, zorder=3)
            ax.annotate(f"{s_}: {y_:.1f}", (x_, top), xytext=(5, -2), textcoords="offset points",
                        fontsize=6 if compact else 7, color="0.35", ha="left", va="top")
        ax.text(j, top * 1.05, f"{v[j]:.2f}\n{star(r.pvalues[k])}", ha="center", va="bottom", fontsize=fs_lab)
    for q, ((k, e, z, p), padj) in enumerate(zip(con, ph)):
        j = X.index(k)
        y = top * (1.42 + 0.19 * q)
        ax.plot([0, 0, j, j], [y - 0.03 * top, y, y, y - 0.03 * top], color="k", lw=1.1, clip_on=False)
        ptxt = "p < 0.001" if padj < 0.001 else f"p = {padj:.3f}"
        ax.text(j / 2, y + 0.015 * top, f"{ptxt} {star(padj)}", ha="center", va="bottom", fontsize=fs_br)
    ax.set_ylim(min(0, per.to_numpy(float).min()) - 0.04 * top, top * (1.42 + 0.19 * len(con) + 0.08))
    ax.axhline(0, ls="-", lw=1, color="k")
    ax.set_xticks(range(K))
    ax.set_xticklabels([LAB[k] for k in X], rotation=25, ha="right", fontsize=fs_tick)
    ax.set_ylabel("replay rate: deviance explained\nlost on dropping (pp)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
