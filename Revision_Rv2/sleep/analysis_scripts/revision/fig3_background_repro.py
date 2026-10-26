"""fig3_background_repro.py -- Figure 3e-h: panels e, f, h reproduced from their stored
inputs with the figure notebook's own functions; panel e redrawn with the background d2
beside the full-window d2, and panel g (CCK silencing) with the background d2 as session
means.

Fig. 3e relates d2 to barrage rate; with the full window, the spikes of the barrages and ripples inside
the window are part of the d2 being correlated. The background d2 (2024 Euclidean, AR 20, 40 ms bins)
excises every ripple and barrage interval and accumulates the autocovariance within each event-free
segment (d2_yw_matlab_match.d2_yw_segments); CA1 NREM, non-overlapping 30 s windows, 12 barrage sessions.
Panels f (branching ratio) and h (ISI entropy) carry no d2.

Inputs: r16_glue_control_windows.csv (column d2_segments); the Fig. 3g rows written by fig3g_background.py
(run it first); function definitions from first_submission_figures/Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb,
executed via ast (the notebook itself is not run).

NONOVERLAP=1 draws every panel on non-overlapping data (f: the *_br_changes_non_overlap.pkl files;
g: rows with centres >= one window apart; h: windows >= 30 s apart, SEM over sessions).

    -> data/revision/log/fig3_background_repro{,_non-overlapping}.csv
    -> reports/revision_figs/fig3_background_repro{,_non-overlapping}.png / .svg
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import ast
import glob
import json
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                          # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
NB = os.path.join(REPO, "first_submission_figures", "Fig3_E_F_G_H_barrage_d2_S2_E_abblate_bar.ipynb")
RB = os.path.join(REPO, "data", "ripple_barr")
LOG = os.path.join(REPO, "data", "revision", "log")
NONOVERLAP = os.environ.get("NONOVERLAP", "0") == "1"
# BINS=width: bin e and h into 6 equal-width bins of each session's barrage-rate range [0, max]
# instead of rank sextiles, so zero-rate windows all sit in bin 1 rather than split by time order.
# Per-panel default: e rank sextiles, h equal-width bins; BINS=width puts both panels on equal width
# (the _equalwidth variant).
WIDTH = os.environ.get("BINS", "rank") == "width"
WIDTH_E, WIDTH_H = WIDTH, True
_SFX = ("_non-overlapping" if NONOVERLAP else "") + ("_equalwidth" if WIDTH else "")
WORD = {True: "equal-width bin", False: "sextile"}
BIN_WORD = WORD[WIDTH_E]


def width_tiles(rate, k=6):
    """k equal-width bins of [0, session max]; the max falls in bin k."""
    r = pd.to_numeric(rate, errors="coerce")
    mx = r.max()
    if not np.isfinite(mx) or mx <= 0:
        return pd.Series(np.nan, index=r.index)
    return pd.Series(np.clip(np.floor(r / mx * k) + 1, 1, k), index=r.index)


def tiles(rate, k=6, width=False):
    return width_tiles(rate, k) if width else _force_sextiles(rate, k=k)                  # noqa: F821
OUT_CSV = os.path.join(LOG, f"fig3_background_repro{_SFX}.csv")
OUT_FIG = os.path.join(REPO, "reports", "revision_figs", f"fig3_background_repro{_SFX}")
BG = os.path.join(LOG, "r16_glue_control_windows.csv")
G_WIN = os.path.join(LOG, "fig3g_background_windows.csv")                 # fig3g_background.py

_cells = json.load(open(NB, encoding="utf-8"))["cells"]
for _i in (1, 9):
    _tree = ast.parse("".join(_cells[_i]["source"]))
    _tree.body = [n for n in _tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
    exec(compile(_tree, NB, "exec"), globals())


def sextile_trend(df, sess_col, value_col, rate_col, z=True):
    """Notebook panel-e steps on a window table (z=False: raw units, no z-score):
    per-session z of the value, within-session
    sextiles of the rate, per-session sextile mean, mean +- SEM over sessions, linregress r."""
    rows = []
    for s, g in df.groupby(sess_col):
        g = g[[value_col, rate_col]].replace([np.inf, -np.inf], np.nan).dropna()
        sd = g[value_col].std(ddof=0)
        if len(g) < 6 or not np.isfinite(sd) or sd == 0:
            continue
        g = g.assign(z=((g[value_col] - g[value_col].mean()) / sd) if z else g[value_col],
                     tile=tiles(g[rate_col], k=6, width=WIDTH_E))
        rows.append(g.groupby("tile").z.mean().rename(s))
    # tile x session: sort by bin, so the line joins the bins in order (a session's missing bin
    # would otherwise come last)
    pf = pd.concat(rows, axis=1).sort_index()
    m, e = pf.mean(axis=1), pf.std(axis=1, ddof=1) / np.sqrt(pf.notna().sum(axis=1))
    ok = np.isfinite(m.to_numpy())
    lr = stats.linregress(pf.index.to_numpy(float)[ok], m.to_numpy()[ok])
    print(f"     {value_col}: sessions per {BIN_WORD} {pf.notna().sum(axis=1).astype(int).tolist()}")
    return m, e, lr, pf.shape[1]


def session_rho(df, sess_col, value_col, rate_col):
    r = df.groupby(sess_col).apply(lambda g: stats.spearmanr(g[value_col], g[rate_col], nan_policy="omit")[0]).dropna()
    return r, stats.wilcoxon(r).pvalue


def nonoverlap(g, win=30.0):
    keep, last = [], -np.inf
    for i, c in zip(g.index, g.center_s.to_numpy()):
        if c >= last + win:
            keep.append(i); last = c
    return g.loc[keep]


def panel_e_published():
    df = load_pool_rate_binned(os.path.join(RB, "mice_d2_barr_ripple"), rate_col="barrage_rate_hz",  # noqa: F821
                               state="NREM", n_bins=6)
    pf = df.groupby(["file_id", "tile"]).d2_norm.mean().unstack("file_id")
    m = pf.mean(axis=1)
    return stats.linregress(pf.index.to_numpy(float), m.to_numpy()), pf.shape[1]


def panel_f(nonoverlap_files=False):
    rows = []
    pat = "*_br_changes_non_overlap.pkl" if nonoverlap_files else "*_br_changes.pkl"
    for f in sorted(glob.glob(os.path.join(RB, "branching_ratio", "barr", pat))):
        p = pickle.load(open(f, "rb"))
        r = p["results"] if isinstance(p, dict) and "results" in p else p
        a, d = os.path.basename(f).split("_")[:2]
        for key, lab in (("barrage_br_changes", "Barrage"), ("barrage_br_changes_shuffle", "Barrage_Shuffle")):
            for v in r.get(key, []):
                rows.append(dict(animal=a, day=d, condition=lab, value=float(v)))
    df = pd.DataFrame(rows)
    df["animal_day"] = df.animal + "_" + df.day
    df = df.dropna().reset_index(drop=True)
    import statsmodels.formula.api as smf
    res = smf.mixedlm("value ~ C(condition) + C(animal)", data=df, groups=df["animal_day"], re_formula="1").fit(method="lbfgs")
    term = "C(condition)[T.Barrage_Shuffle]"
    pf = df.groupby(["animal_day", "condition"]).value.mean().unstack("condition").dropna()
    return pf, float(res.pvalues[term]), float(-res.params[term])


def panel_h(nonoverlap_windows=False, rank_tiles=False, raw=False):
    """rank_tiles: `add_session_tiles` drops tied quantile bins and renumbers from 1, so with most
    windows at barrage rate 0 the top sextiles hold few sessions; rank_tiles uses panel e's
    `_force_sextiles` per session instead (qcut if it yields 6 bins, else equal-count bins on the
    rank, ties in time order), so every session has all six."""
    d = load_pooled_per_center(os.path.join(RB, "ISI_entropy", "barrage"), state_filter="NREM")  # noqa: F821
    d["sess"] = d.animal_name.astype(str) + "_" + d.day.astype(str)
    no = pd.concat([nonoverlap(g.sort_values("center_s")) for _, g in d.groupby("sess")])
    if nonoverlap_windows:
        d = no.reset_index(drop=True)
    if rank_tiles:
        d = d.sort_values(["sess", "center_s"]).reset_index(drop=True)
        d["tile_barrage_sess"] = pd.concat([tiles(g.barrage_rate, k=6, width=WIDTH_H)
                                            for _, g in d.groupby("sess", sort=False)]).astype(float)
    else:
        d = add_session_tiles(d, by="barrage", k=6)                                               # noqa: F821
    d = add_session_zscores(d, "isi_entropy", robust=False)                                       # noqa: F821
    if raw:                                    # ISI entropy without z-score either, as in panel e
        d["isi_entropy_z"] = d["isi_entropy"]  # column name kept for the notebook aggregator
    t = aggregate_from_precomputed_tiles(d, "tile_barrage_sess", "isi_entropy_z", "barrage_rate",  # noqa: F821
                                         error_mode="sessions" if nonoverlap_windows else "windows")
    r, p = stats.pearsonr(t.tile_barrage_sess, t.isi_entropy_z_mean)
    rho, prho = session_rho(no, "sess", "isi_entropy", "barrage_rate")
    return t, r, p, rho, prho, d.sess.nunique()


def main():
    out = []
    # ---- published panel statistics -----------------------------------------------------------
    lr_e, n_e = panel_e_published()
    pf_f, p_f, diff_f = panel_f()
    t_h, r_h, p_h, rho_h, prho_h, n_h = panel_h()
    print("STAGE 1 -- published pipeline on the stored inputs")
    print(f"  e  d2 (full window) z vs barrage sextile: r = {lr_e.rvalue:.3f}, p = {lr_e.pvalue:.4f}, "
          f"{n_e} files   [published r = -0.969, p = 0.0014]")
    print(f"  f  BR barrage - shuffle: LMM p = {p_f:.3g}, diff {diff_f:+.4f}, {len(pf_f)} files   "
          f"[published p = 2.71e-7, n = 11]")
    print(f"  h  ISI entropy z vs barrage sextile: r = {r_h:.3f}, p = {p_h:.4f}, {n_h} sessions   "
          f"[published r = 0.946, p = 0.0043]")
    out += [dict(panel="e_published", stat="sextile_r", value=lr_e.rvalue, p=lr_e.pvalue, n=n_e),
            dict(panel="f_published", stat="lmm_diff", value=diff_f, p=p_f, n=len(pf_f)),
            dict(panel="h_published", stat="sextile_r", value=r_h, p=p_h, n=n_h)]
    if NONOVERLAP:
        pf_f, p_f, diff_f = panel_f(nonoverlap_files=True)
        print("NON-OVERLAPPING -- f from *_br_changes_non_overlap.pkl, g and h on windows >= one window apart")
        print(f"  f  LMM p = {p_f:.3g}, diff {diff_f:+.4f}, {len(pf_f)} files")
        out.append(dict(panel="f_nonoverlap", stat="lmm_diff", value=diff_f, p=p_f, n=len(pf_f)))
    # panel h is drawn with rank sextiles in both modes
    t_h, r_h, p_h, rho_h, prho_h, n_h = panel_h(nonoverlap_windows=NONOVERLAP, rank_tiles=True, raw=True)
    print(f"  h  {WORD[WIDTH_H]}s: r = {r_h:.3f}, p = {p_h:.4f}; sessions per {WORD[WIDTH_H]} "
          f"{t_h.n_sessions.astype(int).tolist()}")
    out.append(dict(panel="h_rank" + ("_nonoverlap" if NONOVERLAP else ""), stat="sextile_r", value=r_h, p=p_h, n=n_h))

    # ---- panel e with the background d2 -------------------------------------------------------
    W = pd.read_csv(BG)
    res = {}
    for col, lab in (("d2_segments", "background"), ("d2_intact", "full window")):
        m, e, lr, n = sextile_trend(W, "session", col, "barrage_rate", z=False)
        rho, prho = session_rho(W, "session", col, "barrage_rate")
        res[lab] = (m, e, lr, n, rho, prho)
        print(f"  e  {lab:11s} d2 (raw) vs barrage sextile: r = {lr.rvalue:.3f}, p = {lr.pvalue:.4f}; per-session "
              f"Spearman {rho.mean():+.3f} (Fisher-z mean {np.tanh(np.arctanh(rho).mean()):+.3f}), "
              f"{int((rho < 0).sum())}/{len(rho)} negative, signed-rank p = {prho:.4f}; {n} sessions, {len(W)} windows")
        out += [dict(panel=f"e_{lab}", stat="sextile_r", value=lr.rvalue, p=lr.pvalue, n=n),
                dict(panel=f"e_{lab}", stat="session_rho_mean", value=float(np.tanh(np.arctanh(rho).mean())),
                     p=prho, n=len(rho), n_dir=int((rho < 0).sum()))]
    wf = stats.wilcoxon(pf_f.Barrage, pf_f.Barrage_Shuffle)
    up_f = int((pf_f.Barrage > pf_f.Barrage_Shuffle).sum())
    print(f"  f  file means, paired Wilcoxon: {up_f}/{len(pf_f)} barrage > shuffle, p = {wf.pvalue:.4f}")
    print(f"  h  per-session Spearman (non-overlapping): Fisher-z mean {np.tanh(np.arctanh(rho_h).mean()):+.3f}, "
          f"{int((rho_h > 0).sum())}/{len(rho_h)} positive, signed-rank p = {prho_h:.4f}")
    out += [dict(panel="f_sessions", stat="wilcoxon", value=float((pf_f.Barrage - pf_f.Barrage_Shuffle).median()),
                 p=wf.pvalue, n=len(pf_f), n_dir=up_f),
            dict(panel="h_sessions", stat="session_rho_mean", value=float(np.tanh(np.arctanh(rho_h).mean())),
                 p=prho_h, n=len(rho_h), n_dir=int((rho_h > 0).sum()))]
    GW = pd.read_csv(G_WIN)
    if NONOVERLAP:
        GW = pd.concat([nonoverlap(g.sort_values("center_s"), 30.0 if s == "SNCG06_day27" else 60.0)
                        for (s, _), g in GW.groupby(["animal_day", "epoch"])])
        print(f"  g  non-overlapping rows: {len(GW)} of {len(pd.read_csv(G_WIN))}")
    G = GW.groupby(["animal_day", "epoch"]).d2_bg.mean().unstack("epoch")
    wg = stats.wilcoxon(G.postsleep, G.presleep)
    up_g = int((G.postsleep > G.presleep).sum())
    print(f"  g  background d2, session means: {G.presleep.mean():.4f} -> {G.postsleep.mean():.4f}, "
          f"{up_g}/{len(G)} up, paired Wilcoxon p = {wg.pvalue:.4f}")
    out.append(dict(panel="g_background_sessions", stat="wilcoxon", value=float((G.postsleep - G.presleep).mean()),
                    p=wg.pvalue, n=len(G), n_dir=up_g))
    pd.DataFrame(out).to_csv(assert_local_path(OUT_CSV), index=False)

    # panel h's statistic: the mixed model of fig3h_entropy_diagnostics.py -- ISI entropy ~ barrage
    # rate + C(animal), random intercept and slope per session, non-overlapping windows (run that
    # script first)
    L = pd.read_csv(os.path.join(LOG, "fig3h_entropy_diagnostics_lmm.csv"))
    L = L[L.panel.str.startswith("h") & (L.model == "C(animal) + (1 + rate|session)")].iloc[0]
    lmm_h = f"{L.coef_per_0p1Hz:+.3f} bits per 0.1 Hz, p = {L.p:.3f}"
    print(f"  h  LMM: {lmm_h}")

    # ---- figure ------------------------------------------------------------------------------
    plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    # e: both arms -- grey = full window (the published quantity), tab:blue = background (ripples and
    # barrages excised); same windows, raw units, session counts per arm
    stars = lambda p: "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
    # colours: background keeps the panel's tab:blue; the added full-window line is grey
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
    print(f"-> {OUT_CSV}\n-> {OUT_FIG}.png/.svg")


if __name__ == "__main__":
    main()
