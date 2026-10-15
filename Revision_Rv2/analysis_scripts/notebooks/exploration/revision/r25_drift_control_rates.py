"""
r25_drift_control_rates.py -- d2 ~ SWR-rate and d2 ~ barrage-rate relations under
detrending, partialling on the within-window trend, and in low-drift windows.

A linear trend inside a 30 s NREM window depresses d2 and correlates with d2 within session.
This script computes the two event-rate relationships (d2 rises with ripple rate, falls with
barrage rate) on the same windows, with session as the unit, under four arms:

  arms   real        d2 (2024 Euclidean, p=20, 40 ms, OLS)
         detrended   the same window with its linear trend removed before fitting
         partial     Spearman r(d2, rate) after residualising both on |trend| (rank-based)
         low-drift   windows with |trend| < 0.5 SD end-to-end only
  plus   r(|trend|, rate): the correlation between within-window drift magnitude and
         event rate

  cohort A  the Fig-1 sessions (CA1, firingRate > 0.2 Hz), ripple rate
  cohort B  the Fig-3 barrage sessions (CA1 units), ripple rate AND barrage rate (barrage
            rate from barrage_sweep_windows.csv; ripple starts from load_ripples_events)

Windows: NREM, 30 s, non-overlapping (make_windows), GUARD_S at every state boundary;
event rate = starts inside the window / 30 s (the convention of the barrage sweep).
Statistics: per-session Spearman r, Fisher-z mean +- SE across sessions, signed-rank,
count of sessions with the published sign.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/r25_drift_control_rates_windows.csv
    -> data/revision/log/r25_drift_control_rates.csv          (per cohort x relationship x arm x session)
    -> reports/revision_figs/r25_drift_control_rates.{png,svg}
"""
from __future__ import annotations

import argparse
import os
import sys
import time
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
from fig1_metrics import (GUARD_S, MIN_FIRING_RATE, binned_counts,          # noqa: E402
                          build_state_intervals, make_windows, slice_sorted)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id  # noqa: E402
from euclid_d2_err_orders import d2_and_sd                                 # noqa: E402
from barrage_order_kernel_sweep import BARRAGE_SESSIONS                    # noqa: E402
from r25_stationarity_battery import detrend, drift_metrics, zs            # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
BARR_WIN = os.path.join(LOGDIR, "barrage_sweep_windows.csv")
BIN = 0.040
P = 20
WIN_S = 30.0
LOW_DRIFT_SD = 0.5
MIN_WIN = 10
ARMS = ["real", "detrended", "partial", "low-drift"]


def _starts(bp, loader):
    from neuro_py.io import loading
    try:
        ev = getattr(loading, loader)(basepath=bp)
    except Exception:                                            # noqa: BLE001
        return None
    if ev is None or not len(ev) or "start" not in getattr(ev, "columns", []):
        return None
    return np.sort(np.asarray(ev["start"], float))


def _rate(ev, w0, w1):
    if ev is None:
        return np.nan
    return (np.searchsorted(ev, w1, "right") - np.searchsorted(ev, w0, "left")) / (w1 - w0)


def one_session(args):
    cohort, name, bp = args
    from neuro_py.io import loading
    try:
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        iv = build_state_intervals(loading.load_SleepState_states(bp), GUARD_S)
    except Exception as e:                                        # noqa: BLE001
        return cohort, name, None, f"{type(e).__name__}: {e}"
    if "NREM" not in iv or not len(iv["NREM"]):
        return cohort, name, None, "no NREM"
    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    if keep.sum() < 5:
        return cohort, name, None, "too few units"
    sp = np.sort(np.concatenate([np.asarray(u, float)
                                 for u in np.asarray(st.data, dtype=object)[keep]]))
    rip = _starts(bp, "load_ripples_events")
    barr = _starts(bp, "load_barrage_events") if cohort == "barrage" else None
    win = make_windows(iv["NREM"], "NREM")
    win = win[win.nonoverlap].reset_index(drop=True)
    rows = []
    for r in win.itertuples(index=False):
        seg = slice_sorted(sp, r.win_start, r.win_stop)
        x = binned_counts(seg, r.win_start, r.win_stop, BIN, presliced=True)
        if x.size < P + 20 or x.std() == 0:
            continue
        o = d2_and_sd(zs(x), P)
        if o is None:
            continue
        xd = detrend(x)
        od = d2_and_sd(zs(xd), P) if xd.std() > 0 else None
        trend, half = drift_metrics(x)
        rows.append(dict(cohort=cohort, session=name, w=int(round(r.win_start)),
                         d2_real=o[0], d2_detrended=np.nan if od is None else od[0],
                         trend_sd=trend, halfdiff_sd=half, n_units=int(keep.sum()),
                         rate_hz=float(x.mean() / BIN),
                         ripple_rate=_rate(rip, r.win_start, r.win_stop),
                         barrage_rate_ev=_rate(barr, r.win_start, r.win_stop)))
    W = pd.DataFrame(rows)
    return cohort, name, W, ""


def _partial_spearman(a, b, c):
    """Spearman r(a, b | c): correlate the residuals of rank(a) and rank(b) on rank(c)."""
    ra, rb, rc = (stats.rankdata(v) for v in (a, b, c))
    A = np.column_stack([np.ones(len(rc)), rc])
    ea = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
    eb = rb - A @ np.linalg.lstsq(A, rb, rcond=None)[0]
    return float(stats.pearsonr(ea, eb)[0])


def per_session_stats(W, rate_col):
    out = {}
    v = W[["d2_real", "d2_detrended", "trend_sd", rate_col]].dropna()
    if len(v) < MIN_WIN or v[rate_col].std() == 0:
        return None
    tr = v.trend_sd.abs()
    out["real"] = float(stats.spearmanr(v.d2_real, v[rate_col]).statistic)
    out["detrended"] = float(stats.spearmanr(v.d2_detrended, v[rate_col]).statistic)
    out["partial"] = _partial_spearman(v.d2_real.to_numpy(), v[rate_col].to_numpy(), tr.to_numpy())
    lo = v[tr < LOW_DRIFT_SD]
    out["low-drift"] = (float(stats.spearmanr(lo.d2_real, lo[rate_col]).statistic)
                        if len(lo) >= MIN_WIN and lo[rate_col].std() > 0 else np.nan)
    out["trend_vs_rate"] = float(stats.spearmanr(tr, v[rate_col]).statistic)
    out["trend_vs_d2"] = float(stats.spearmanr(tr, v.d2_real).statistic)
    out["n_windows"] = len(v)
    out["frac_low_drift"] = float((tr < LOW_DRIFT_SD).mean())
    return out


def fisher(v):
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)])
    if v.size == 0:
        return np.nan, np.nan, 0
    z = np.arctanh(np.clip(v, -0.999999, 0.999999))
    return float(np.tanh(z.mean())), float(z.std(ddof=1) / np.sqrt(z.size)) if z.size > 1 else np.nan, z.size


RELS = [("fig1", "ripple_rate", "Fig 2 cohort: d2 ~ ripple rate", +1),
        ("barrage", "ripple_rate", "Fig 3 cohort: d2 ~ ripple rate", +1),
        ("barrage", "barrage_rate", "Fig 3 cohort: d2 ~ barrage rate", -1)]


def report(R):
    print("\n" + "=" * 112)
    print("d2 (2024 Euclidean, p=20, 40 ms) ~ event rate, NREM 30 s non-overlapping windows; session = unit")
    print("arms: real | linearly detrended d2 | partial r given |trend| | windows with |trend| < 0.5 SD only")
    print("=" * 112)
    for cohort, rate_col, label, sign in RELS:
        P_ = R[(R.cohort == cohort) & (R.rate_col == rate_col)]
        if not len(P_):
            continue
        print(f"\n  {label}  (n = {P_.session.nunique()} sessions)")
        print(f"  {'arm':12s}{'Fisher r':>10s}{'± SE':>8s}{'sign-consistent':>17s}{'signed-rank p':>15s}")
        for arm in ARMS + ["trend_vs_rate", "trend_vs_d2"]:
            v = P_[arm].to_numpy(float)
            v = v[np.isfinite(v)]
            if v.size < 3:
                continue
            fr, se, n = fisher(v)
            cons = int((np.sign(v) == sign).sum()) if arm in ARMS else int((v < 0).sum())
            lab = arm if arm in ARMS else ("|trend| ~ rate" if arm == "trend_vs_rate" else "|trend| ~ d2")
            note = "" if arm in ARMS else ("  (neg count)")
            print(f"  {lab:12s}{fr:+10.3f}{se:8.3f}{cons:>9d}/{n:<7d}{stats.wilcoxon(v).pvalue:15.4f}{note}")


def window_length_panel(ax, rng, title):
    """d: the ripple-rate relation at shorter (and longer) windows, computed on the window
    table in window_length_sweep_windows.csv (NREM, non-overlapping, d2 (2024 Euclidean,
    p = 20, 40 ms) by Yule-Walker, all CA1 units), per-session Spearman as panels a-c."""
    W = pd.read_csv(os.path.join(LOGDIR, "window_length_sweep_windows.csv"))
    W = W[W.state == "NREM"].dropna(subset=["d2ex", "ripple_rate"])
    lengths = sorted(W.window_s.unique())
    for i, L in enumerate(lengths):
        v = np.array([stats.spearmanr(h.d2ex, h.ripple_rate).statistic
                      for _, h in W[W.window_s == L].groupby(["animal", "day"])
                      if len(h) >= 10 and h.ripple_rate.nunique() > 2])
        ax.plot(i + rng.normal(0, 0.05, v.size), v, "o", ms=4, color="#7b3294", mec="k", mew=0.4, alpha=0.85)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=2.5)
        ax.text(i, 0.62, f"{int((v > 0).sum())}/{v.size}\np={stats.wilcoxon(v).pvalue:.3f}", ha="center", fontsize=7.5)
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.axvspan(-0.5, 1.5, color="0.92", zorder=0)
    ax.text(0.5, -0.55, "N/p < 20", ha="center", fontsize=7.5, color="0.4")
    ax.set_xticks(range(len(lengths))); ax.set_xticklabels([f"{int(L)} s" for L in lengths], fontsize=8)
    ax.set_xlim(-0.5, len(lengths) - 0.5)
    ax.set_xlabel("window length (non-overlapping)", fontsize=8.5)
    ax.set_title(title, fontsize=9.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def drift_panel(ax, R, cohort, rate_col, title, sign, rng, col):
    P_ = R[(R.cohort == cohort) & (R.rate_col == rate_col)]
    for i, arm in enumerate(ARMS):
        v = P_[arm].to_numpy(float)
        v = v[np.isfinite(v)]
        ax.plot(i + rng.normal(0, 0.05, v.size), v, "o", ms=4, color=col[arm], mec="k", mew=0.4, alpha=0.85)
        ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color="k", lw=2.5)
        pw = stats.wilcoxon(v).pvalue if v.size > 2 else np.nan
        ax.text(i, 0.62, f"{int((np.sign(v) == sign).sum())}/{v.size}\np={pw:.3f}", ha="center", fontsize=7.5)
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels(["d2 as\npublished", "d2\ndetrended", "partial r\n(| trend)", "low-drift\nwindows"], fontsize=8)
    ax.set_title(title, fontsize=9.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def plot(R, full=False):
    """a = SWR-rate relation by window length, b = SWR-rate relation under drift control
    (Fig-1 cohort), c = barrage-rate relation under drift control (barrage cohort); the
    barrage cohort's SWR-rate panel is dropped. full=True redraws the four-panel
    version (a-c drift control incl. that panel, d window length) to backup/."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 10, "axes.linewidth": 1.2})
    col = {"real": "#7b3294", "detrended": "#9970ab", "partial": "#c2a5cf", "low-drift": "#e7d4e8"}
    rng = np.random.default_rng(0)
    if full:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1, 1.15]})
        for ax, (cohort, rate_col, label, sign), letter in zip(axes, RELS, "abc"):
            drift_panel(ax, R, cohort, rate_col, f"{letter}  {label}", sign, rng, col)
        window_length_panel(axes[3], rng, "d  Fig 2 cohort: d2 ~ ripple rate by window length")
        out_dir, name = os.path.join(FIGDIR, "backup"), "r25_drift_control_rates_4panel"
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.8), sharey=True, gridspec_kw={"width_ratios": [1.15, 1, 1]})
        window_length_panel(axes[0], rng, "a  d2 ~ SWR rate by window length")
        drift_panel(axes[1], R, "fig1", "ripple_rate", "b  d2 ~ SWR rate, drift controlled", +1, rng, col)
        drift_panel(axes[2], R, "barrage", "barrage_rate", "c  d2 ~ barrage rate, drift controlled", -1, rng, col)
        out_dir, name = FIGDIR, "r25_drift_control_rates"
    axes[0].set_ylabel("per-session Spearman r")
    axes[0].set_ylim(-0.6, 0.75)
    fig.tight_layout()
    os.makedirs(assert_local_path(out_dir), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(out_dir, f"{name}.{ext}")), dpi=300)
    print(f"\nfigure -> {os.path.join(out_dir, name)}.png/.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    win_csv = os.path.join(LOGDIR, "r25_drift_control_rates_windows.csv")
    out_csv = os.path.join(LOGDIR, "r25_drift_control_rates.csv")
    if args.report_only:
        R = pd.read_csv(out_csv)
        report(R)
        plot(R)
        plot(R, full=True)
        return
    jobs = [("fig1", "_".join(session_id(bp)), bp) for bp in FIG1_SESSIONS]
    jobs += [("barrage", n, bp) for n, bp in BARRAGE_SESSIONS.items()]
    t0 = time.time()
    from multiprocessing import Pool
    tabs = []
    with Pool(processes=args.workers) as pool:
        for cohort, name, W, err in pool.imap_unordered(one_session, jobs):
            print(f"  {cohort:8s} {name:13s} {err or f'{len(W):5d} windows'}", flush=True)
            if W is not None and len(W):
                tabs.append(W)
    W = pd.concat(tabs, ignore_index=True)
    # barrage rate: from the barrage sweep's window table (CA1 pop), keyed on w
    B = pd.read_csv(BARR_WIN)
    B = B[B["pop"] == "CA1"][["session", "w", "barrage_rate"]]
    W = W.merge(B, on=["session", "w"], how="left")
    W["barrage_rate"] = W["barrage_rate"].fillna(W["barrage_rate_ev"])
    assert_local_path(LOGDIR)
    W.to_csv(assert_local_path(win_csv), index=False)
    rows = []
    for cohort, rate_col, label, sign in RELS:
        for s, g in W[W.cohort == cohort].groupby("session"):
            st_ = per_session_stats(g, rate_col)
            if st_ is None:
                continue
            rows.append(dict(cohort=cohort, rate_col=rate_col, session=s, **st_))
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(out_csv), index=False)
    print(f"\n{len(W)} windows, {len(R)} session x relationship rows in {time.time() - t0:.0f} s")
    report(R)
    plot(R)
    plot(R, full=True)


if __name__ == "__main__":
    main()
