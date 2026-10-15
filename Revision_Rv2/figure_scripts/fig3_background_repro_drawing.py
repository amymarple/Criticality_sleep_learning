"""Drawing functions from analysis_scripts/notebooks/exploration/revision/ (fig3_background_repro.py, run_fig1_metrics.py) used by the
figure notebooks. LOG, OUT_FIG, OUT_CSV: the folder of the notebook's tables, set by the notebook.
"""
import matplotlib.pyplot as plt
import os

LOG = None
OUT_FIG = None
OUT_CSV = None


# --- fig3_background_repro.py
NONOVERLAP = os.environ.get("NONOVERLAP", "0") == "1"


# --- fig3_background_repro.py
WIDTH = os.environ.get("BINS", "rank") == "width"


# --- fig3_background_repro.py
WIDTH_E, WIDTH_H = WIDTH, True


# --- fig3_background_repro.py
WORD = {True: "equal-width bin", False: "sextile"}


# --- fig3_background_repro.py
BIN_WORD = WORD[WIDTH_E]


# --- run_fig1_metrics.py
WRITABLE_DRIVES = {"C", "D"}


# --- run_fig1_metrics.py
def assert_local_path(path):
    """Raise unless `path` is on a local writable drive.

    The PreToolUse hook cannot see writes made *inside* a running process, so the
    guarantee that this script never touches the ayadata shares is enforced here.
    Called before every write.
    """
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    drive = os.path.splitdrive(p)[0].rstrip(":").upper()
    if drive not in WRITABLE_DRIVES:
        raise PermissionError(
            f"refusing to write outside {sorted(WRITABLE_DRIVES)}: {p} "
            "(raw session data is read-only — see DATA_SAFETY.md)"
        )
    return p


# --- fig3_background_repro.py (a drawing section, as a function)
def draw_main(res, pf_f, p_f, up_f, wf, G, up_g, wg, t_h, r_h, p_h, lmm_h):
    """The figure section of main() in fig3_background_repro.py (from \'# ---- figure\' on), drawn from the variables computed before it: res, pf_f, p_f, up_f, wf, G, up_g, wg, t_h, r_h, p_h, lmm_h."""
    # ---- figure ------------------------------------------------------------------------------
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    # e: both arms -- grey = full window (the published quantity), tab:blue = background (ripples and
    # barrages excised); same windows, raw units, session counts per arm
    stars = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
    # colours: background keeps the panel's original tab:blue; the added full-window line is grey
    for k, (lab, col, name) in enumerate((("full window", "#7f7f7f", "full window"),
                                          ("background", "tab:blue", "ripples + barrages excised"))):
        m, e, lr, n, rho, prho = res[lab]
        x = m.index.to_numpy(int)
        ax[0].errorbar(x, m, yerr=e, fmt="-o", capsize=4, color=col, lw=2, ms=5, label=name, zorder=3 if k else 2)
        ax[0].text(0.97, 0.96 - 0.09 * k, f"{int((rho < 0).sum())}/{len(rho)} ↓ {stars(prho)}", transform=ax[0].transAxes,
                   ha="right", va="top", color=col, fontsize=9)
    ax[0].set_xticks(x); ax[0].set_xlabel(f"Barrage rate {BIN_WORD} (low → high)")
    ax[0].set_ylabel("d2 (2024 Euclidean, AR 20), raw")
    ax[0].legend(frameon=False, fontsize=8, loc="lower left")
    rf, rb = res["full window"], res["background"]
    ax[0].set_title(f"e  d2 vs barrage rate, {rb[3]} sessions\nr (sextile means): full {rf[2].rvalue:.3f}, "
                    f"background {rb[2].rvalue:.3f}", loc="left", fontsize=9)
    order = ["Barrage", "Barrage_Shuffle"]
    for i, c in enumerate(order):
        ax[1].boxplot(pf_f[c], positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                      boxprops=dict(facecolor=("tab:blue", "tab:cyan")[i], alpha=0.8), medianprops=dict(color="k"))
    for _, r in pf_f.iterrows():
        ax[1].plot([0, 1], [r.Barrage, r.Barrage_Shuffle], "-", color="gray", alpha=0.6, lw=1.1)
        ax[1].scatter([0, 1], [r.Barrage, r.Barrage_Shuffle], s=14, color="k", zorder=3)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["BARR", "shuffle"]); ax[1].set_ylabel("Δ branching ratio")
    ax[1].set_title(f"f  {'non-overlapping' if NONOVERLAP else '(as published)'}\nLMM p = {p_f:.2g}; sessions {up_f}/{len(pf_f)}, p = {wf.pvalue:.4f}",
                    loc="left", fontsize=9)
    pal = {"presleep": "tab:cyan", "postsleep": "tab:blue"}
    for i, c in enumerate(("presleep", "postsleep")):
        ax[2].boxplot(G[c], positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                      boxprops=dict(facecolor=pal[c], alpha=0.8), medianprops=dict(color="k"))
    for _, r in G.iterrows():
        ax[2].plot([0, 1], [r.presleep, r.postsleep], "-", color="gray", alpha=0.6, lw=1.1)
        ax[2].scatter([0, 1], [r.presleep, r.postsleep], s=14, color="k", zorder=3)
    ax[2].set_xticks([0, 1]); ax[2].set_xticklabels(["Baseline", "Barrage\ndisruption"])
    ax[2].set_ylabel("background d2 (session mean)")
    ax[2].set_title(f"g  CCK silencing, background d2\nsessions {up_g}/{len(G)} up, paired Wilcoxon p = {wg.pvalue:.4f}",
                    loc="left", fontsize=9)
    ax[3].errorbar(t_h.tile_barrage_sess, t_h.isi_entropy_z_mean, yerr=t_h.isi_entropy_z_sem, marker="o", lw=2,
                   capsize=4, color="tab:blue")
    ax[3].set_xticks(t_h.tile_barrage_sess.astype(int)); ax[3].set_xlabel(f"Barrage rate {WORD[WIDTH_H]} (low → high)")
    ax[3].set_ylabel("ISI entropy (bits)")
    ax[3].set_title(f"h  {'non-overlapping, ' if NONOVERLAP else ''}{'equal-width bins' if WIDTH_H else 'rank sextiles'}\n6 bin means: r = {r_h:.3f}, p = {p_h:.3f}\n"
                    f"LMM (animal; random session slope): {lmm_h}", loc="left", fontsize=9)
    fig.text(0.01, -0.04, ("Non-overlapping windows throughout (f: *_br_changes_non_overlap.pkl; g: windows >= 60 s "
             "apart, 30 s for SNCG06_day27; h: >= 30 s apart, SEM over sessions). " if NONOVERLAP else "") + "NREM. e: d2 (2024 Euclidean, AR 20, 40 ms bins, CA1), 30 s non-overlapping windows, grey the full window, blue with every "
             "ripple and barrage interval excised (within-segment Yule–Walker); 12 sessions; raw units (no z-score, e and h); mean ± SEM "
             "over sessions. g: Sncg sessions, the submitted rows (CA1+CA2, 80 ms bins, 60 s windows; SNCG06_day27 40 ms / "
             "30 s), ripples and barrages excised, mean per session and epoch. f, h: the submitted panels recomputed "
             "from their stored inputs. e: rank sextiles within session; h: 6 equal-width bins of each "
             "session's barrage-rate range [0, max]" + (" (e too)" if WIDTH_E else "") + ". " +
             "Session statistics: "
             "per-session Spearman or file means, signed-rank across sessions; h: linear mixed model, ISI entropy ~ barrage "
             "rate + animal (fixed), random intercept and slope per session, non-overlapping windows.", fontsize=7.5, color="#555555", wrap=True)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{OUT_FIG}.{ext}"), dpi=300, bbox_inches="tight")
