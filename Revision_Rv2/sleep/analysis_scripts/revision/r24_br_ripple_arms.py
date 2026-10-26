"""
r24_br_ripple_arms.py -- the branching ratio m against SWR rate, full window and ripple-excised, next
to d2 (2024 Euclidean) on the same windows (Extended Data Fig. 4c, d), and where each estimator's input
sits relative to the ripples.

Both m arms are computed on the 16 Fig-1 sessions with a ripple file, on the windows of `d2_arms_raw.py`,
which holds both d2 arms for these sessions.

WINDOWS. `d2_arms_raw_windows.csv`, state NREM: Fig 1's non-overlapping 30 s tiles, 16 sessions
(those with a ripple file), 5 404 windows, with d2_full / d2_excised (all CA1
units, 40 ms, AR(20); validated there against the pickle and *_rippleablate.mat) and ripple_rate.
  m   `excised_br_entropy.window_metrics` as is: MR estimator (fit_m), pyramidal active-unit
      counts (rate > 0.2 Hz), 4 ms, k_max = 200 (800 ms); full = one segment, excised = the ripple-
      free sub-windows (>= 0.1 s, >= 1 s in total), r_k pooled with no lag crossing a gap.
      Checked against the Fig-1 pickle's BR_pyr_k200 (full) and excised_br_entropy's BR_excised
      (the 8 sessions in common).
STATS. Session is the unit: per-session Spearman with SWR rate, Fisher-z mean +- SEM, sessions
positive, signed-rank; sextiles within session in raw units.

IN-RIPPLE SHARES (same pass). For every window, the share of each estimator's input autocovariance
that comes from bin pairs with at least one bin inside a ripple -- the pairs excision deletes --
against the share of bins inside ripples:
    m   pyramidal active-unit count, 4 ms; lags 1-10 (4-40 ms, where r_k is largest) and 1-200
    d2  all-CA1 spike count, 40 ms; lags 1-20 (the AR(20) Yule-Walker range)
share_K = 1 - sum_{k in K} C_free(k) / sum_{k in K} C(k), C(k) = sum_t d_t d_{t+k} on the
mean-removed window, C_free the same over pairs with both bins outside every ripple (FFT).
PASS=0 re-plots from the stored table.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/r24_br_ripple_windows.csv    (per window: both m arms + shares)
    -> data/revision/log/r24_br_ripple_arms.csv       (per arm statistics, shares)
    -> reports/revision_figs/backup/r24_br_ripple_arms.png / .svg   (4 panels; Extended Data Fig. 4c, d
       are drawn by r24_figure.py with sextile_panel())
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
from run_fig1_metrics import assert_local_path, FIG1_SESSIONS, session_id    # noqa: E402
from fig1_metrics import active_unit_counts, BIN_BR_S, MIN_FIRING_RATE        # noqa: E402
from d2_yw_matlab_match import getFR                                         # noqa: E402
from r35_glm_ablation import LOGDIR, FIGDIR, WIN_S                           # noqa: E402
from excised_br_entropy import free_segments, window_metrics, MIN_FREE_S     # noqa: E402
from d2_arms_raw import sextile_curve, fisher, stars, X6, GREY, RED, PKL_DIR  # noqa: E402

D2_WIN = os.path.join(LOGDIR, "d2_arms_raw_windows.csv")
BR_REF = os.path.join(LOGDIR, "excised_br_entropy_windows.csv")
# POP=all: m on all CA1 units (rate > 0.2 Hz), i.e. the population d2 uses -- the matched comparator.
# Default: pyramidal.
POP = os.environ.get("POP", "pyr")
_SFX = "" if POP == "pyr" else f"_{POP}"
OUT_WIN = os.path.join(LOGDIR, f"r24_br_ripple_windows{_SFX}.csv")
OUT_SUM = os.path.join(LOGDIR, f"r24_br_ripple_arms{_SFX}.csv")
BR_PICKLE_COL = "BR_pyr_k200" if POP == "pyr" else "BR_all_k200"
ARMS = [("d2_full", "d2, full window"), ("d2_excised", "d2, ripple-excised"),
        ("m_full", "m, full window"), ("m_excised", "m, ripple-excised")]
D2_DT, D2_P = 0.04, 20
D2_YLAB = "d2 (2024 Euclidean, p = 20, 40 ms)"
M_YLAB = f"m (MR, {'pyramidal' if POP == 'pyr' else 'all CA1 units'}, 4 ms, k$_{{max}}$ = 200)"


def ripple_mask(t0, dt, n, r0, r1):
    """bin i = [t0 + i dt, t0 + (i+1) dt) overlaps any ripple [r0, r1) (sorted, non-overlapping)"""
    e0 = t0 + dt * np.arange(n); e1 = e0 + dt
    j = np.searchsorted(r1, e0, "right")
    ok = j < r0.size
    m = np.zeros(n, bool)
    m[ok] = r0[j[ok]] < e1[ok]
    return m


def share(x, mask, lags):
    d = np.asarray(x, float); d = d - d.mean()
    f = (~mask).astype(float)
    nfft = 1 << int(np.ceil(np.log2(2 * d.size)))
    kmax = max(b for _, b in lags.values())

    def ac(y):
        F = np.fft.rfft(y, nfft)
        return np.fft.irfft(F * np.conj(F), nfft)[:kmax + 1]
    C, Cf = ac(d), ac(d * f)
    out = {}
    for name, (a, b) in lags.items():
        s = C[a:b + 1].sum()
        out[name] = 1 - Cf[a:b + 1].sum() / s if s > 0 else np.nan
    return out


def m_pass(W):
    from neuro_py.io import loading
    done = pd.read_csv(OUT_WIN) if os.path.exists(OUT_WIN) else pd.DataFrame()
    rows = done.to_dict("records") if len(done) else []
    want = set(W.session)
    for bp in FIG1_SESSIONS:
        a, d = session_id(bp); sid = f"{a}_{d}"
        if sid not in want or (len(done) and sid in set(done.session)):
            continue
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        units_all = [np.sort(np.asarray(s, float)) for s in np.asarray(st.data, dtype=object)]
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        ctype = np.asarray(cm.putativeCellType)
        units = [u for u, k in zip(units_all, keep) if k]
        e_spikes = [u for u, k, c in zip(units_all, keep, ctype) if k and (POP == "all" or "Pyramidal" in str(c))]
        all_sorted = np.sort(np.concatenate(units))                     # as excised_br_entropy
        sp_all = np.sort(np.concatenate(units_all)); sp_all = sp_all[np.isfinite(sp_all)]   # as the d2 producer
        rip = loading.load_ripples_events(bp)
        r0 = np.asarray(rip["start"], float); o = np.argsort(r0)
        r0, r1 = r0[o], np.asarray(rip["stop"], float)[o]
        n0 = len(rows)
        for c in W.loc[W.session == sid, "center_s"].to_numpy(float):
            t0, t1 = c - WIN_S / 2, c + WIN_S / 2
            m_full, _, _ = window_metrics(e_spikes, all_sorted, [(t0, t1)])
            segs = free_segments(t0, t1, r0, r1)
            m_exc = window_metrics(e_spikes, all_sorted, segs)[0] if sum(b - a for a, b in segs) >= MIN_FREE_S else np.nan
            A = active_unit_counts(e_spikes, t0, t1, BIN_BR_S); mA = ripple_mask(t0, BIN_BR_S, A.size, r0, r1)
            X = getFR(sp_all, t0, t1, D2_DT); mX = ripple_mask(t0, D2_DT, X.size, r0, r1)
            rec = dict(session=sid, center_s=c, m_full=m_full, m_excised=m_exc,
                       time_share_4ms=mA.mean(), time_share_40ms=mX.mean())
            rec.update({f"m_{k}": v for k, v in share(A, mA, {"var": (0, 0), "lag4_40ms": (1, 10), "lag4_800ms": (1, 200)}).items()})
            rec.update({f"d2_{k}": v for k, v in share(X, mX, {"var": (0, 0), "lag40_800ms": (1, D2_P)}).items()})
            rows.append(rec)
        S = pd.DataFrame(rows[n0:])
        msg = f"  {sid:12s} {len(S):4d} windows"
        pk = os.path.join(PKL_DIR, f"{sid}_CA1_fig1_metrics_nonoverlap.pkl")
        if os.path.exists(pk):
            P = pickle.load(open(pk, "rb"))["windows"][["center_s", BR_PICKLE_COL]]
            v = S.merge(P, on="center_s").dropna(subset=["m_full", BR_PICKLE_COL])
            msg += f" | r(m_full, pickle {BR_PICKLE_COL}) = {np.corrcoef(v.m_full, v[BR_PICKLE_COL])[0, 1]:.4f}"
        R = pd.read_csv(BR_REF)                                   # excised reference exists for pyramidal only
        v = S.merge(R[R.session == sid][["center_s", "BR_excised"]], on="center_s").dropna(subset=["m_excised", "BR_excised"])
        if len(v) > 2 and POP == "pyr":
            msg += f" | r(m_excised, excised_br_entropy) = {np.corrcoef(v.m_excised, v.BR_excised)[0, 1]:.4f} (n = {len(v)})"
        msg += (f" | in-SWR share: bins {100 * S.time_share_4ms.median():.1f} %, m 4-40 ms {100 * S.m_lag4_40ms.median():.0f} %, "
                f"d2 40-800 ms {100 * S.d2_lag40_800ms.median():.0f} %")
        print(msg, flush=True)
        pd.DataFrame(rows).to_csv(assert_local_path(OUT_WIN), index=False)
    return pd.DataFrame(rows)


def n_dir(t):
    """sessions in the direction of the mean effect"""
    return t["n_pos"] if t["r"] > 0 else t["n"] - t["n_pos"]


def load_table(run_pass=False):
    """NREM windows with both d2 arms (d2_arms_raw) joined to both m arms (this script's pass)."""
    W = pd.read_csv(D2_WIN)
    W = W[W.state == "NREM"][["session", "center_s", "d2_full", "d2_excised", "ripple_rate"]]
    M = m_pass(W) if run_pass or not os.path.exists(OUT_WIN) else pd.read_csv(OUT_WIN)
    return W.merge(M, on=["session", "center_s"], how="inner")


def arm_stats(D, verbose=True):
    """per-session Spearman of each arm with SWR rate; Fisher mean +- SEM, sessions positive, signed-rank"""
    rows = []
    for col, lab in ARMS:
        r = np.array([stats.spearmanr(g[col], g.ripple_rate, nan_policy="omit").statistic for _, g in D.groupby("session")])
        m, se, n = fisher(r)
        rec = dict(arm=col, label=lab, r=m, se=se, n_pos=int((r > 0).sum()), n=n, p=stats.wilcoxon(r).pvalue,
                   per_session=";".join(f"{x:.3f}" for x in r))
        rows.append(rec)
        if verbose:
            print(f"  {lab:20s} {m:+.3f} +- {se:.3f}  {rec['n_pos']}/{n}  p = {rec['p']:.4f}")
    return rows


def sextile_panel(ax, D, T, full, exc, ylab, title, legend_loc="lower right", count_y=0.97):
    """raw values by within-session SWR-rate sextile, full (grey) vs ripple-excised (red)"""
    for k, (col, c, lab) in enumerate(((full, GREY, "full window"), (exc, RED, "ripple-excised"))):
        m, e, n = sextile_curve(D, col, "ripple_rate")
        ax.errorbar(X6, m, yerr=e, fmt="o-", capsize=4, lw=2, ms=5, color=c, label=lab, zorder=3 - k)
        t = T[col]
        ax.text(0.03, count_y - 0.09 * k, f"{n_dir(t)}/{t['n']} {'↑' if t['r'] > 0 else '↓'} {stars(t['p'])}",
                transform=ax.transAxes, va="top", color=c, fontsize=9)
    ax.set_xticks(X6); ax.set_xlabel("SWR rate sextile (within session)"); ax.set_ylabel(ylab + "\nraw, mean ± SEM over sessions")
    ax.legend(frameon=False, fontsize=8, loc=legend_loc)
    ax.set_title(title, loc="left", fontweight="bold")


def main():
    D = load_table(run_pass=os.environ.get("PASS") != "0")
    print(f"\nper-session Spearman with SWR rate, NREM, {D.session.nunique()} sessions (Fisher mean +- SEM, sessions positive, signed-rank)")
    rows = arm_stats(D)
    Sm = D.groupby("session").median(numeric_only=True)
    print("\nin-ripple share, per-session median, mean +- SEM over sessions")
    for c in ("time_share_4ms", "m_var", "m_lag4_40ms", "m_lag4_800ms", "time_share_40ms", "d2_var", "d2_lag40_800ms"):
        print(f"  {c:16s} {100 * Sm[c].mean():5.1f} +- {100 * Sm[c].sem():.1f} %")
        rows.append(dict(arm="share", label=c, r=Sm[c].mean(), se=Sm[c].sem(), n=len(Sm)))
    print(f"\nlevel: m full {Sm.m_full.mean():.3f} +- {Sm.m_full.sem():.3f}, excised {Sm.m_excised.mean():.3f} +- {Sm.m_excised.sem():.3f}; "
          f"tau full {(-4 / np.log(Sm.m_full)).mean():.1f} ms, excised {(-4 / np.log(Sm.m_excised)).mean():.1f} ms")
    pd.DataFrame(rows).to_csv(assert_local_path(OUT_SUM), index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.7), gridspec_kw={"width_ratios": [1, 1, 1.05, 1.15]})
    T = {r["arm"]: r for r in rows if r["arm"] != "share"}
    sextile_panel(axes[0], D, T, "d2_full", "d2_excised", D2_YLAB, "a  d2 vs SWR rate")
    sextile_panel(axes[1], D, T, "m_full", "m_excised", M_YLAB, "b  branching ratio m vs SWR rate",
                  legend_loc="upper right", count_y=0.14)          # m declines: counts go bottom-left
    ax = axes[2]
    xl = ["d2\nfull", "d2\nexcised", "m\nfull", "m\nexcised"]
    for i, (col, _) in enumerate(ARMS):
        t = T[col]; v = np.array([float(x) for x in t["per_session"].split(";")])
        c = RED if "excised" in col else GREY
        ax.scatter(i + np.random.default_rng(i).uniform(-0.12, 0.12, v.size), v, s=14, color=c, alpha=0.55, lw=0)
        ax.errorbar(i, t["r"], yerr=t["se"], fmt="s", color="k", ms=6, capsize=4, zorder=4)
        ax.text(i, 0.58, f"{n_dir(t)}/{t['n']} {'↑' if t['r'] > 0 else '↓'}\n{stars(t['p'])}", ha="center", va="top", fontsize=8.5,
                fontweight="bold" if t["p"] < .05 else "normal")
    ax.axhline(0, color="k", lw=0.8, ls="--"); ax.set_xticks(range(4)); ax.set_xticklabels(xl)
    ax.set_ylim(-0.55, 0.6); ax.set_ylabel("Spearman ρ with SWR rate (per session)")
    ax.set_title("c  per session", loc="left", fontweight="bold")
    ax = axes[3]
    bars = [("time_share_4ms", "SWR bins,\n4 ms", "#bdbdbd"), ("m_lag4_40ms", "m: lags\n4–40 ms", "#3182bd"),
            ("m_lag4_800ms", "m: lags\n4–800 ms", "#9ecae1"), ("time_share_40ms", "SWR bins,\n40 ms", "#bdbdbd"),
            ("d2_lag40_800ms", "d2: lags\n40–800 ms", "#756bb1")]
    for i, (c, lab, col) in enumerate(bars):
        v = Sm[c].to_numpy() * 100
        ax.bar(i, v.mean(), yerr=v.std(ddof=1) / np.sqrt(v.size), color=col, width=0.7, capsize=3, edgecolor="k", lw=0.6)
        ax.scatter(i + np.random.default_rng(10 + i).uniform(-0.15, 0.15, v.size), v, s=10, color="k", alpha=0.5, lw=0, zorder=3)
        ax.text(i, max(v.max(), v.mean() + v.std(ddof=1) / np.sqrt(v.size)) + 2, f"{v.mean():.0f} %", ha="center", fontsize=8.5)
    ax.axvline(2.5, color="k", lw=0.6, ls=":")
    ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[1] for b in bars], fontsize=8)
    ax.set_ylabel("share inside SWRs (%)\nper-session median, mean ± SEM")
    ax.set_title("d  where each estimator's signal sits", loc="left", fontweight="bold")
    fig.suptitle(f"NREM, {D.session.nunique()} sessions, non-overlapping 30 s windows; grey = full window, "
                 "red = ripple-excised; session is the unit", fontsize=9, y=1.01)
    fig.tight_layout()
    # the 4-panel figure goes to backup/; r24_figure.py draws the SWR-rate panels with sextile_panel()
    os.makedirs(assert_local_path(os.path.join(FIGDIR, "backup")), exist_ok=True)
    out = os.path.join(FIGDIR, "backup", f"r24_br_ripple_arms{_SFX}.png")
    fig.savefig(assert_local_path(out), dpi=200, bbox_inches="tight"); fig.savefig(out[:-4] + ".svg", bbox_inches="tight")
    print("  ->", out)


if __name__ == "__main__":
    main()
