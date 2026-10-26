"""
d2_arms_raw.py -- full-window and ripple-excised d2 (2024 Euclidean, p = 20, 40 ms) in raw units, across the four
states and against ripple rate, barrage rate and replay rate. No z-scoring anywhere: the full-window arm is the d2
of the .mat files (Fig. 1); the ripple-excised arm removes the in-ripple spikes that couple d2 to population rate.

ARMS. Both computed with the MATLAB producer's estimator (`d2_yw_matlab_match.d2_yw_segments`:
getFR 40 ms bins, pooled z-score, within-segment Yule-Walker autocovariance, |1 - sum phi| /
sqrt(p)); all CA1 cells, as the .mat files. Full = the whole 30 s window (checked against
the Fig-1 pickle's `d2`, which is the *_error_bar.mat value). Excised = the window minus
every ripple interval, sub-windows >= 0.1 s, at least MIN_FREE_S of free time (checked
against *_rippleablate.mat for NREM -- the .mat only ablates NREM (isNREM guard), so the
REM / THETA / nonTHETA excised values exist only here).

COHORTS.
  states                the Fig-1 sessions with a ripple table (16), all four states: the non-overlapping
                        _error_bar.mat windows of r24_state_centre.py, wake split by the window centre
                        (LABEL=centre; LABEL=wholewindow: the Fig-1 tiles, whole window inside the state)
  ripple rate           the same sessions, NREM, non-overlapping tiles
                        (`data/revision/*_fig1_metrics_nonoverlap.pkl` + share spikes/ripples)
  barrage rate          12 barrage sessions, NREM, from r16_glue_control_windows.csv:
                        d2_intact and d2_segments (the same within-segment estimator with
                        BOTH ripples and barrages excised)
  replay rate           9 decoded sessions, NREM decoded time, excised_br_entropy_windows.csv
                        (d2_published / d2_excised = the .mat arms; replay rate = replay
                        SWRs per s = replay_prob x ripple_rate)
STATS. Session is the unit. States: per-session median per state, paired Wilcoxon against NREM and within sleep
(NREM vs REM) and within wake (non-theta vs theta), the last two drawn. Rates: per-session Spearman, signed-rank
across sessions; sextiles within session, mean +- SEM over sessions in raw units.

Reads the shares read-only; writes only to D:. FIG1_ONLY=1 skips the share pass if the
windows CSV exists (plots from the stored table).
    -> data/revision/log/d2_arms_raw_windows.csv     (Fig-1 cohort, both arms, per tile)
    -> data/revision/log/d2_arms_raw_windows_centre.csv  (LABEL=centre: states panel, .mat windows;
                                                          Extended Data Fig. 2h)
    -> data/revision/log/d2_arms_raw_summary.csv     (LABEL=wholewindow: *_summary_wholewindow.csv)
    -> reports/revision_figs/d2_arms_raw.png / .svg  (LABEL=wholewindow: backup/d2_arms_raw_wholewindow.png)
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path, FIG1_SESSIONS, session_id   # noqa: E402
from r35_glm_ablation import LOGDIR, FIGDIR, WIN_S                          # noqa: E402
from d2_yw_matlab_match import d2_yw_segments                               # noqa: E402
from excised_br_entropy import free_segments, MIN_FREE_S                    # noqa: E402
from fig1_d2_vs_mua_2x2 import nrem_mat                                     # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
PKL_DIR = os.path.join(REPO, "data", "revision")
OUT_WIN = os.path.join(LOGDIR, "d2_arms_raw_windows.csv")
OUT_SUM = os.path.join(LOGDIR, "d2_arms_raw_summary.csv")
# LABEL=centre (default): the STATES panel uses the wake split of Fig 1D -- a window is THETA if its centre lies
# in SleepState THETA -- on the non-overlapping _error_bar.mat windows of r24_state_centre.py, both arms
# recomputed here. LABEL=wholewindow uses the Fig-1 tiles (whole window inside THETA / nonTHETA) instead.
LABEL = os.environ.get("LABEL", "centre")
CENTRE_SRC = os.path.join(LOGDIR, "r24_state_centre_windows.csv")
OUT_WIN_C = os.path.join(LOGDIR, "d2_arms_raw_windows_centre.csv")
if LABEL != "centre":
    OUT_SUM = OUT_SUM.replace(".csv", f"_{LABEL}.csv")
STATES = ["NREM", "REM", "THETA", "nonTHETA"]
STATE_LABEL = {"NREM": "NREM", "REM": "REM", "THETA": "theta wake", "nonTHETA": "non-theta wake"}
XPOS = np.array([0, 1, 2.5, 3.5])
PAIRS = [("NREM", "REM"), ("nonTHETA", "THETA")]
GREY, RED, BLUE = "#7f7f7f", "#d62728", "#1f77b4"   # BLUE: the events-excised arm of the barrage-rate panel
X6 = np.arange(1, 7)


def stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


# ----------------------------------------------------------------------------- Fig-1 cohort
def fig1_pass():
    """Per session, checkpointed: sessions already in OUT_WIN are skipped."""
    from neuro_py.io import loading
    done = pd.read_csv(OUT_WIN) if os.path.exists(OUT_WIN) else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    for bp in FIG1_SESSIONS:
        animal, day = session_id(bp)
        sid = f"{animal}_{day}"
        if len(done) and sid in set(done.session):
            print(f"  {sid:12s} from checkpoint"); continue
        pk = os.path.join(PKL_DIR, f"{animal}_{day}_CA1_fig1_metrics_nonoverlap.pkl")
        if not os.path.exists(pk):
            print(f"  [SKIP] {sid}: no pickle"); continue
        W = pickle.load(open(pk, "rb"))["windows"]
        W = W[W["nonoverlap"].astype(bool)]
        st, _ = loading.load_spikes(bp, brainRegion="CA1")
        sp = np.sort(np.concatenate([np.asarray(u, float).ravel() for u in np.asarray(st.data, dtype=object)]))
        sp = sp[np.isfinite(sp)]
        rip = loading.load_ripples_events(bp)
        if rip is None or not hasattr(rip, "columns") or "start" not in rip.columns or len(rip) == 0:
            print(f"  [SKIP] {sid}: ripple table unusable ({None if rip is None else list(getattr(rip, 'columns', []))[:5]})", flush=True)
            continue
        r0 = np.asarray(rip["start"], float); o = np.argsort(r0)
        r0, r1 = r0[o], np.asarray(rip["stop"], float)[o]
        for r in W.itertuples(index=False):
            t0, t1 = r.center_s - WIN_S / 2, r.center_s + WIN_S / 2
            segs = free_segments(t0, t1, r0, r1)
            free = sum(b - a for a, b in segs)
            n_rip = int(np.searchsorted(r0, t1, "left") - np.searchsorted(r0, t0, "left"))
            rows.append(dict(session=sid, state=r.state, center_s=r.center_s, d2_pickle=r.d2, MUA_mean=r.MUA_mean,
                             d2_full=d2_yw_segments(sp, [(t0, t1)]),
                             d2_excised=d2_yw_segments(sp, segs) if free >= MIN_FREE_S else np.nan,
                             ripple_rate=n_rip / WIN_S, frac_removed=1 - free / WIN_S))
        D = pd.DataFrame([x for x in rows if x["session"] == sid])
        v = D[["d2_full", "d2_pickle"]].dropna(); r_full = np.corrcoef(v.T.to_numpy())[0, 1] if len(v) > 2 else np.nan
        msg = f"  {sid:12s} {len(D):4d} tiles  r(full YW, pickle) = {r_full:.4f}"
        M = nrem_mat(bp, "_rippleablate")
        if M is not None:
            N = D[D.state == "NREM"].merge(M.rename(columns={"v": "mat"}), on="center_s", how="inner")
            v = N[["d2_excised", "mat"]].dropna()
            if len(v) > 2:
                msg += f"  r(excised YW, _rippleablate.mat) = {np.corrcoef(v.T.to_numpy())[0, 1]:.4f} (n = {len(v)})"
        print(msg + f"  ripples removed: median {100 * D.frac_removed.median():.1f} % of the NREM tile", flush=True)
        pd.DataFrame(rows).to_csv(assert_local_path(OUT_WIN), index=False)          # checkpoint
    D = pd.DataFrame(rows)
    D.to_csv(assert_local_path(OUT_WIN), index=False)
    return D


def centre_pass():
    """Both arms on the centre-labelled non-overlapping .mat windows (all four states), checkpointed."""
    from neuro_py.io import loading
    C = pd.read_csv(CENTRE_SRC)
    done = pd.read_csv(OUT_WIN_C) if os.path.exists(OUT_WIN_C) else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    for bp in FIG1_SESSIONS:
        animal, day = session_id(bp); sid = f"{animal}_{day}"
        g = C[C.session == sid]
        if not len(g) or (len(done) and sid in set(done.session)):
            continue
        rip = loading.load_ripples_events(bp)
        if rip is None or not hasattr(rip, "columns") or "start" not in rip.columns or len(rip) == 0:
            print(f"  [SKIP] {sid}: no ripple table", flush=True); continue
        st, _ = loading.load_spikes(bp, brainRegion="CA1")
        sp = np.sort(np.concatenate([np.asarray(u, float).ravel() for u in np.asarray(st.data, dtype=object)]))
        sp = sp[np.isfinite(sp)]
        r0 = np.asarray(rip["start"], float); o = np.argsort(r0)
        r0, r1 = r0[o], np.asarray(rip["stop"], float)[o]
        n0 = len(rows)
        for r in g.itertuples(index=False):
            t0, t1 = r.center - WIN_S / 2, r.center + WIN_S / 2
            segs = free_segments(t0, t1, r0, r1)
            free = sum(b - a for a, b in segs)
            n_rip = int(np.searchsorted(r0, t1, "left") - np.searchsorted(r0, t0, "left"))
            rows.append(dict(session=sid, state=r.state, center_s=r.center, d2_mat=r.d2,
                             d2_full=d2_yw_segments(sp, [(t0, t1)]),
                             d2_excised=d2_yw_segments(sp, segs) if free >= MIN_FREE_S else np.nan,
                             ripple_rate=n_rip / WIN_S, frac_removed=1 - free / WIN_S))
        D = pd.DataFrame(rows[n0:])
        v = D[["d2_full", "d2_mat"]].dropna()
        print(f"  {sid:12s} {len(D):4d} centre-labelled windows  r(full YW, .mat) = {np.corrcoef(v.T.to_numpy())[0, 1]:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(assert_local_path(OUT_WIN_C), index=False)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- summaries
def fisher(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    z = np.arctanh(np.clip(v, -0.999, 0.999))
    return float(np.tanh(z.mean())), float(np.tanh(z.std(ddof=1) / np.sqrt(len(z)))) if len(z) > 1 else np.nan, len(v)


def rate_summary(W, d2col, ratecol, label, arm, rows, partial=None):
    rs, ps = [], []
    for s, g in W.groupby("session"):
        v = g[[d2col, ratecol] + ([partial] if partial else [])].dropna()
        if len(v) < 10 or v[ratecol].nunique() <= 2:
            continue
        rs.append(stats.spearmanr(v[d2col], v[ratecol]).statistic)
        if partial:
            from r35_fig2d_replot import partial_spearman
            ps.append(partial_spearman(v[d2col].to_numpy(), v[ratecol].to_numpy(), v[partial].to_numpy()))
    r, se, n = fisher(rs); p = stats.wilcoxon(rs).pvalue if len(rs) > 2 else np.nan
    rec = dict(block=label, arm=arm, r=r, se=se, n_pos=int((np.array(rs) > 0).sum()), n=n, p=p)
    if partial:
        rp, sep, _ = fisher(ps); rec.update(r_partial=rp, p_partial=stats.wilcoxon(ps).pvalue if len(ps) > 2 else np.nan)
    rows.append(rec)
    return rec


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


def main():
    if os.environ.get("FIG1_ONLY") and os.path.exists(OUT_WIN):
        F = pd.read_csv(OUT_WIN)
    else:
        F = fig1_pass()
    B = pd.read_csv(os.path.join(LOGDIR, "r16_glue_control_windows.csv"))
    B = B.rename(columns={"d2_intact": "d2_full", "d2_segments": "d2_excised"})
    R = pd.read_csv(os.path.join(LOGDIR, "excised_br_entropy_windows.csv"))
    R = R.rename(columns={"d2_published": "d2_full"}).dropna(subset=["replay_prob"])
    R["replay_rate"] = R["replay_prob"] * R["ripple_rate"]
    rows = []

    # -- states: per-session median per state, both arms
    if LABEL == "centre":
        Fs = pd.read_csv(OUT_WIN_C) if os.environ.get("FIG1_ONLY") and os.path.exists(OUT_WIN_C) else centre_pass()
        Fs = Fs[Fs.session.isin(set(F.session))]          # the 16 sessions with a ripple file, as panels b-d
    else:
        Fs = F
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

    # -- rates
    print("\nRATES -- per-session Spearman (Fisher mean), sessions positive, signed-rank p")
    blocks = [("ripple rate (Fig-1 NREM)", F[F.state == "NREM"], "ripple_rate", None),
              ("barrage rate (barrage cohort)", B, "barrage_rate", "ripple_rate"),
              ("ripple rate (barrage cohort)", B, "ripple_rate", "barrage_rate"),
              ("replay rate (decoded 9)", R, "replay_rate", "ripple_rate")]
    for label, W, col, partial in blocks:
        for arm in ("d2_full", "d2_excised"):
            rec = rate_summary(W, arm, col, label, arm, rows, partial)
            extra = f"   | partial on {partial}: {rec['r_partial']:+.3f} p={rec['p_partial']:.4f}" if partial else ""
            print(f"  {label:32s} {arm:11s} r = {rec['r']:+.3f} +- {rec['se']:.3f}  {rec['n_pos']}/{rec['n']}  p = {rec['p']:.4f}{extra}")
    pd.DataFrame(rows).to_csv(assert_local_path(OUT_SUM), index=False)

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
    for ax, (label, W, col, xlab, letter) in zip(axes[1:], [
            ("ripple rate", F[F.state == "NREM"], "ripple_rate", "SWR rate sextile (NREM)", "b"),
            ("barrage rate", B, "barrage_rate", "barrage rate sextile (NREM)", "c"),
            ("replay rate", R, "replay_rate", "replay-SWR rate sextile (NREM, decoded time)", "d")]):
        # barrage-rate panel: the events-excised arm in blue
        exc_col = BLUE if letter == "c" else RED
        for arm, c, lab in (("d2_full", GREY, "full window"), ("d2_excised", exc_col, "ripple-excised" if letter != "c" else "event-excised")):
            m, e, n = sextile_curve(W, arm, col)
            ax.errorbar(X6, m, yerr=e, fmt="o-", capsize=4, lw=2, ms=5, color=c, label=lab, zorder=3 if arm == "d2_excised" else 2)
            rec = [r for r in rows if r["block"].startswith(label) and r["arm"] == arm][0]
            up = rec["r"] > 0                                           # falling curves: label top right, clear of sextile 1
            ax.text(0.03 if up else 0.97, 0.96 - (0.09 if arm == "d2_excised" else 0), f"{rec['n_pos']}/{rec['n']} {'↑' if up else '↓'} {stars(rec['p'])}",
                    transform=ax.transAxes, va="top", ha="left" if up else "right", color=c, fontsize=9)
        ax.set_xticks(X6); ax.set_xlabel(xlab); ax.set_ylabel("d2 (2024 Euclidean, p = 20), raw")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.set_title(f"{letter}  {label}, {W.session.nunique()} sessions", loc="left", fontweight="bold")
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


if __name__ == "__main__":
    main()
