"""fig2g_psth_excised.py -- Figure 2g in its own design: the peri-ripple branching-ratio dip
vs shuffled times, on the nine decoded sessions, full-window and ripple-excised.

The published Fig 2g (generate_br_change_ripple.ipynb -> Fig2_G_branchingratio.ipynb) is a
PSTH, not a sextile plot: branching ratio (BR) from 10 s windows stepped 2 s (5 s state
guard, 4 ms bins, active-unit counts, MR estimator k_max = 200; meta of
branching_ratio_allstates_win10s_step2s_guard5s_*.pkl) aligned to every NREM ripple peak;
per ripple, change = (mean BR within +/-2.5 s - mean BR in [-20,-17.5] u [17.5,20]) /
baseline; the same for 5x as many random NREM times excluding [start-1, stop+1] of every
ripple; the two per-session means compared paired across sessions. Fig 2's nine decoded
sessions (PPP7_day8) are used here; the BR series is recomputed for all nine.

Arms. full: the published recipe, recomputed. excised: every 10 s window's BR from its
ripple-free sub-windows (raw ripple intervals removed; r_k pooled across sub-windows, no lag crosses a gap;
state_level_br.rk_curve_multi), so the PSTH shows what the background does around a
ripple once the ripples themselves -- the triggering one included -- are out.

    -> data/revision/log/fig2g_psth_nine_sessions.csv      (per session x arm: mean change, shuffle)
    -> data/revision/log/fig2g_psth_curves.csv              (per session x arm x lag bin: PSTH)
    -> reports/revision_figs/fig2g_psth_nine_sessions.png / .svg
"""
from __future__ import annotations

import os
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
from run_fig1_metrics import assert_local_path                                   # noqa: E402
from fig1_metrics import (active_unit_counts, rk_curve, fit_m, build_state_intervals,  # noqa: E402
                          GUARD_S, MIN_FIRING_RATE, BIN_BR_S)
from state_level_br import rk_curve_multi                                        # noqa: E402
from r35_glm_ablation import LOGDIR, FIGDIR                                      # noqa: E402
from r35_swr_property_control import SESSION_TEMPLATE, SESSIONS                  # noqa: E402
from excised_br_entropy import free_segments                                     # noqa: E402

WIN, STEP, K_MAX = 10.0, 2.0, 200
CENTRE = (-2.5, 2.5)
BASE = [(-20.0, -17.5), (17.5, 20.0)]
PSTH = (-20.0, 20.0, 2.0)
N_SHUF = 5
SEED = 0
STORED = os.path.join(os.path.dirname(LOGDIR), "..", "ripple_barr", "branching_ratio", "ripple")
OUT_S = os.path.join(LOGDIR, "fig2g_psth_nine_sessions.csv")
OUT_C = os.path.join(LOGDIR, "fig2g_psth_curves.csv")
SESS = os.environ.get("SESS", "")


def br_series(e_spikes, ivs, r0, r1):
    """BR per 10 s / 2 s window inside the guarded NREM intervals, full and excised."""
    cen, full, exc = [], [], []
    for a, b in ivs:
        t = a
        while t + WIN <= b:
            t0, t1 = t, t + WIN
            k, rk, se = rk_curve(active_unit_counts(e_spikes, t0, t1, BIN_BR_S), K_MAX + 1)
            full.append(fit_m(k, rk, se, K_MAX))
            segs = free_segments(t0, t1, r0, r1)
            if sum(bb - aa for aa, bb in segs) >= 1.0:
                K, rkm, sem, _ = rk_curve_multi([active_unit_counts(e_spikes, aa, bb, BIN_BR_S) for aa, bb in segs], K_MAX + 1)
                exc.append(fit_m(K, rkm, sem, K_MAX))
            else:
                exc.append(np.nan)
            cen.append(t0 + WIN / 2)
            t += STEP
    return np.array(cen), np.array(full), np.array(exc)


def changes(cen, br, events):
    """Per event: relative BR change, centre window vs the two baseline windows."""
    out = np.full(len(events), np.nan)
    for i, e in enumerate(events):
        c = br[(cen >= e + CENTRE[0]) & (cen <= e + CENTRE[1])]
        b = np.concatenate([br[(cen >= e + lo) & (cen <= e + hi)] for lo, hi in BASE])
        c, b = c[np.isfinite(c)], b[np.isfinite(b)]
        if c.size and b.size and b.mean() != 0:
            out[i] = (c.mean() - b.mean()) / b.mean()
    return out


def psth(cen, br, events):
    edges = np.arange(PSTH[0], PSTH[1] + PSTH[2], PSTH[2])
    M = np.full((len(events), len(edges) - 1), np.nan)
    for i, e in enumerate(events):
        rel = cen - e
        for j in range(len(edges) - 1):
            m = (rel >= edges[j]) & (rel < edges[j + 1]) & np.isfinite(br)
            if m.any():
                M[i, j] = br[m].mean()
    # mean over events, and the SEM over events (plotted as the shaded band)
    return edges[:-1] + PSTH[2] / 2, np.nanmean(M, 0), np.nanstd(M, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(M), 0))


EXCLUDE = bool(os.environ.get("EXCLUDE_RIPPLES"))   # default False: see below


def random_times(nrem_iv, r0, r1, ripples, rng):
    """N_SHUF x as many random NREM times as ripples per interval.

    Default: no exclusion window around ripples (the published shuffle). EXCLUDE_RIPPLES=1
    excludes [start-1, stop+1] around every ripple from the draw."""
    out = []
    for a, b in nrem_iv:
        n = int(((ripples >= a) & (ripples < b)).sum()) * N_SHUF
        if n == 0:
            continue
        grid = np.arange(a, b, 0.01)
        bad = np.zeros(grid.size, bool)
        if EXCLUDE:
            lo = np.searchsorted(r1 + 1.0, a); hi = np.searchsorted(r0 - 1.0, b)
            for s, e in zip(r0[lo:hi] - 1.0, r1[lo:hi] + 1.0):
                bad |= (grid >= s) & (grid < e)
        ok = grid[~bad]
        if ok.size:
            out.append(rng.choice(ok, min(n, ok.size), replace=ok.size < n))
    return np.sort(np.concatenate(out)) if out else np.zeros(0)


def main():
    from neuro_py.io import loading
    rng = np.random.default_rng(SEED)
    rows, curves = [], []
    for sess in SESSION_TEMPLATE:
        if SESS and sess != SESS:
            continue
        bp = SESSIONS[sess]
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        units = [np.sort(np.asarray(s, float)) for s in np.asarray(st.data, dtype=object)[keep]]
        ctype = np.asarray(cm.putativeCellType)[keep]
        e_spikes = [u for u, c in zip(units, ctype) if "Pyramidal" in str(c)]
        sd = loading.load_SleepState_states(bp)
        nrem_raw = np.asarray(sd["NREMstate"], float)
        ivs = build_state_intervals(sd, GUARD_S)["NREM"]
        rip = loading.load_ripples_events(bp)
        r0 = np.asarray(rip["start"], float); o = np.argsort(r0)
        r0, r1, peaks = r0[o], np.asarray(rip["stop"], float)[o], np.asarray(rip["peaks"], float)[o]
        in_nrem = np.zeros(peaks.size, bool)
        for a, b in nrem_raw:
            in_nrem |= (peaks >= a) & (peaks <= b)
        ripples = peaks[in_nrem]
        cen, full, exc = br_series(e_spikes, ivs, r0, r1)
        shuf = random_times(nrem_raw, r0, r1, ripples, rng)
        # compare against the stored series where available
        f = os.path.join(STORED, f"branching_ratio_allstates_win10s_step2s_guard5s_{sess}_CA1.pkl")
        note = ""
        if os.path.exists(f):
            o_ = pd.read_pickle(f)["NREM"]
            sc, sb = np.asarray(o_["centers"], float), np.asarray(o_["branching_ratios"], float)
            j = np.searchsorted(sc, cen); j = np.clip(j, 0, sc.size - 1)
            ok = np.abs(sc[j] - cen) < 1e-6
            r = np.corrcoef(full[ok][np.isfinite(full[ok] * sb[j][ok])], sb[j][ok][np.isfinite(full[ok] * sb[j][ok])])[0, 1] if ok.sum() > 10 else np.nan
            note = f" | vs stored series: {int(ok.sum())} matched centres, r = {r:.3f}"
        for arm, br in (("full", full), ("excised", exc)):
            ch_r, ch_s = changes(cen, br, ripples), changes(cen, br, shuf)
            t, p_r, e_r = psth(cen, br, ripples); _, p_s, e_s = psth(cen, br, shuf)
            rows.append(dict(session=sess, arm=arm, n_windows=len(cen), n_ripples=len(ripples),
                             change_ripple=float(np.nanmean(ch_r)), change_shuffle=float(np.nanmean(ch_s)),
                             br_median=float(np.nanmedian(br))))
            for ti, a_, b_, ea, eb in zip(t, p_r, p_s, e_r, e_s):
                curves.append(dict(session=sess, arm=arm, lag_s=ti, br_ripple=a_, br_shuffle=b_,
                                   sem_ripple=ea, sem_shuffle=eb))
        print(f"  {sess:12s} {len(cen):5d} windows, {len(ripples):5d} NREM ripples | full change {rows[-2]['change_ripple']:+.4f} "
              f"vs shuffle {rows[-2]['change_shuffle']:+.4f} | excised {rows[-1]['change_ripple']:+.4f} vs {rows[-1]['change_shuffle']:+.4f}{note}",
              flush=True)
    R, C = pd.DataFrame(rows), pd.DataFrame(curves)
    assert_local_path(LOGDIR)
    suf = f"_{SESS}" if SESS else ""
    R.to_csv(assert_local_path(OUT_S.replace(".csv", f"{suf}.csv")), index=False)
    C.to_csv(assert_local_path(OUT_C.replace(".csv", f"{suf}.csv")), index=False)
    report(R)
    plot(R, C)


def report(R):
    print(f"\n{'=' * 80}\nFig 2g: peri-ripple BR change (centre +/-2.5 s vs baseline at +/-17.5-20 s), "
          f"per-session mean; Ripple vs Shuffle paired across sessions\n{'=' * 80}")
    for arm in ("full", "excised"):
        P = R[R.arm == arm]
        d = P.change_ripple - P.change_shuffle
        print(f"  {arm:8s} N = {len(P)} | ripple {P.change_ripple.median():+.4f}  shuffle {P.change_shuffle.median():+.4f}  "
              f"| ripple < shuffle in {int((d < 0).sum())}/{len(d)}  signed-rank p = {stats.wilcoxon(d).pvalue:.3f}  "
              f"| ripple change < 0 in {int((P.change_ripple < 0).sum())}/{len(P)} (p = {stats.wilcoxon(P.change_ripple).pvalue:.3f})")


def plot(R, C):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5, "legend.fontsize": 10})
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={"width_ratios": [1.4, 1, 1.4, 1]})
    for k, arm in enumerate(("full", "excised")):
        col = "#7f7f7f" if arm == "full" else "#2c7fb8"
        ax = axes[2 * k]
        Cc = C[C.arm == arm]
        g = Cc.groupby("lag_s")
        for key, c_, lab in (("br_ripple", col, "ripple-aligned"), ("br_shuffle", "#e9afc6" if arm == "full" else "#c6dbef", "shuffled times")):
            m, se = g[key].mean(), g[key].std(ddof=1) / np.sqrt(g[key].count())
            ax.plot(m.index, m.values, "-o", ms=4, lw=2, color=c_, label=lab)
            ax.fill_between(m.index, m.values - se.values, m.values + se.values, color=c_, alpha=0.25, lw=0)
        ax.axvline(0, ls="--", lw=1, color="k")
        ax.set_xlabel("time from ripple peak (s)")
        ax.set_ylabel(f"branching ratio m ({arm})")
        ax.set_title(f"{arm} window: PSTH, mean ± SEM over sessions", loc="left", fontsize=11)
        ax.legend(frameon=False)
        ax = axes[2 * k + 1]
        P = R[R.arm == arm].set_index("session")
        x0, x1 = 0, 1
        for s_, r_ in P.iterrows():
            ax.plot([x0, x1], [r_.change_ripple, r_.change_shuffle], "-", color="0.6", lw=1, alpha=0.7)
        ax.plot([x0] * len(P), P.change_ripple, "o", color=col, ms=7, zorder=3)
        ax.plot([x1] * len(P), P.change_shuffle, "o", color="0.55", ms=7, zorder=3)
        d = P.change_ripple - P.change_shuffle
        pw = stats.wilcoxon(d).pvalue
        star = "***" if pw < .001 else "**" if pw < .01 else "*" if pw < .05 else "n.s."
        ax.text(0.5, 0.97, f"{int((d < 0).sum())}/{len(d)} ↓  {star}", ha="center", va="top", transform=ax.transAxes, fontweight="bold")
        ax.axhline(0, ls="--", lw=1, color="k")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["ripple", "shuffle"]); ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("BR change (centre − baseline) / baseline")
        ax.set_title("per session", loc="left", fontsize=11)
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    plt.tight_layout()
    suf = f"_{SESS}" if SESS else ""          # a single-session run must not overwrite the nine-session figure
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"fig2g_psth_nine_sessions{suf}.{ext}")), dpi=300)
    plt.close(fig)
    print(f"  -> {FIGDIR}\\fig2g_psth_nine_sessions{suf}.png/.svg")


if __name__ == "__main__":
    if os.environ.get("REPORT_ONLY"):
        R, C = pd.read_csv(OUT_S), pd.read_csv(OUT_C)
        report(R); plot(R, C)
    else:
        main()
