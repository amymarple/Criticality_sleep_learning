"""
r25_stationarity_battery.py -- stationarity battery: window-length invariance, unit-root
tests, drift surrogates, detrending, and the drift actually present in each window.

The AR fit underlying d2 assumes a stationary window. Slow drift (e.g. an integrated
random walk) pushes sum(phi) -> 1, so d2 -> 0; states can differ in how much slow
structure they contain. The longer the window, the more slow structure it contains, so a
drift-driven d2 falls with window length beyond what finite-sample bias produces. Per
session and state:

  real         d2 (2024 Euclidean, p=20, 40 ms bins, OLS as the MATLAB producer) on
               non-overlapping windows of 10, 15, 20, 30, 60, 120 s.
  detrended    the same windows with a linear trend removed before z-scoring.
  stationary   the finite-sample baseline. An AR(20) fitted on the LONGEST windows
               available (120 s, or the longest the state allows) is simulated with
               fresh Gaussian innovations and cut into every shorter length. Fitting from
               the longest window avoids compounding the short-window bias that a
               per-window own-fit surrogate would carry.
  ramp0.5/1/2  the real z-scored window plus a linear ramp of +-A SD end-to-end.
  rw           plus an integrated random walk scaled to 1 SD over the window.

Every variant also carries the within-window ADF (H0 unit root; maxlag = p = 20,
constant) and KPSS (H0 stationary; nlags = 20, constant) p-values, and every real window
carries the drift actually present in it: the linear trend of the 40 ms counts across
the window in SD units (end-to-end) and the half-window mean change.

Statistics (session = unit): per-session slope of d2 against log2(window length), real
minus stationary, signed-rank; within-session Spearman r of d2 against |trend|; and the
state contrasts recomputed on the lowest-drift tercile of windows. lambda =
1 - mean(sd^2)/var(d2) per state and length. 10 s windows (N/p = 12.5) are below the
N/p >= 20 floor and are shown for the trend only.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/r25_stationarity_windows.csv   (long: one row per window x variant)
    -> data/revision/log/r25_stationarity_summary.csv   (per state x length x variant)
    -> reports/revision_figs/r25_stationarity.{png,svg}
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
from numpy.lib.stride_tricks import sliding_window_view
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from fig1_metrics import (GUARD_S, MIN_FIRING_RATE, STATES, binned_counts,   # noqa: E402
                          build_state_intervals, make_windows, slice_sorted)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id   # noqa: E402
from euclid_d2_err_orders import d2_and_sd                                  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
BIN = 0.040
P = 20
LENGTHS = (10, 15, 20, 30, 60, 120)
RAMPS = (0.5, 1.0, 2.0)
VARIANTS = ["real", "detrended", "stationary"] + [f"ramp{a:g}" for a in RAMPS] + ["rw"]
STATE_COLOR = {"THETA": "#d95f0e", "nonTHETA": "#fdb863", "NREM": "#7b3294", "REM": "#d7b5d8"}
SEED = 20260916


# ----------------------------------------------------------------------------- pieces
def fit_ar(xz, p=P):
    n = xz.size
    if n < p + 20:
        return None
    X = sliding_window_view(xz, p)[:-1][:, ::-1]
    y = xz[p:]
    try:
        phi = np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError:
        return None
    r = y - X @ phi
    return phi, float(r @ r) / max(y.size - p, 1)


def simulate_ar(phi, sig2, n, rng, burn=400):
    from scipy.signal import lfilter
    e = rng.normal(0.0, np.sqrt(max(sig2, 1e-12)), n + burn)
    x = lfilter([1.0], np.concatenate([[1.0], -phi]), e)[burn:]
    if not np.isfinite(x).all() or x.std() == 0:
        return None
    return x


def zs(x):
    return (x - x.mean()) / x.std()


def unit_root_tests(x):
    from statsmodels.tsa.stattools import adfuller, kpss
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            p_adf = float(adfuller(x, maxlag=P, regression="c", autolag=None)[1])
        except Exception:                                # noqa: BLE001
            p_adf = np.nan
        try:
            p_kpss = float(kpss(x, regression="c", nlags=P)[1])
        except Exception:                                # noqa: BLE001
            p_kpss = np.nan
    return p_adf, p_kpss


def drift_metrics(x):
    n = x.size
    sd = x.std()
    if sd == 0:
        return np.nan, np.nan
    slope = np.polyfit(np.arange(n), x, 1)[0]
    half = n // 2
    return float(slope * (n - 1) / sd), float((x[half:].mean() - x[:half].mean()) / sd)


def measure(xz):
    out = d2_and_sd(xz, P)
    if out is None:
        return None
    d2, sd, s, _ = out
    p_adf, p_kpss = unit_root_tests(xz)
    return dict(d2=d2, sd=sd, sumphi=s, p_adf=p_adf, p_kpss=p_kpss)


def detrend(x):
    t = np.arange(x.size)
    return x - np.polyval(np.polyfit(t, x, 1), t)


# ----------------------------------------------------------------------------- session
def one_session(bp):
    from neuro_py.io import loading
    animal, day = session_id(bp)
    rng = np.random.default_rng(SEED + (hash((animal, day)) % 10_000))
    try:
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        sd_ = loading.load_SleepState_states(bp)
    except Exception as e:                                        # noqa: BLE001
        return animal, day, None, f"{type(e).__name__}: {e}"
    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    sp = np.sort(np.concatenate(
        [np.asarray(u, float) for u in np.asarray(st.data, dtype=object)[keep]]))
    iv = build_state_intervals(sd_, GUARD_S)
    rows = []

    def add(state, L, w, variant, m, extra=None):
        rec = dict(animal=animal, day=day, state=state, L=L, w=float(w), variant=variant)
        rec.update(m)
        if extra:
            rec.update(extra)
        rows.append(rec)

    for state in STATES:
        if state not in iv or not len(iv[state]):
            continue
        # ---- real, detrended, drift surrogates at every length
        for L in LENGTHS:
            win = make_windows(iv[state], state, window=float(L), step=float(L))
            for r in win.itertuples(index=False):
                seg = slice_sorted(sp, r.win_start, r.win_stop)
                x = binned_counts(seg, r.win_start, r.win_stop, BIN, presliced=True)
                if x.size < P + 20 or x.std() == 0:
                    continue
                xz = zs(x)
                trend, halfdiff = drift_metrics(x)
                m = measure(xz)
                if m is None:
                    continue
                add(state, L, r.win_start, "real", m,
                    dict(N=int(x.size), rate_hz=float(x.mean() / BIN),
                         trend_sd=trend, halfdiff_sd=halfdiff))
                xd = detrend(x)
                if xd.std() > 0:
                    md = measure(zs(xd))
                    if md is not None:
                        add(state, L, r.win_start, "detrended", md)
                t = np.linspace(-1.0, 1.0, xz.size)
                for a in RAMPS:
                    mr = measure(zs(xz + a * t))
                    if mr is not None:
                        add(state, L, r.win_start, f"ramp{a:g}", mr)
                rw = np.cumsum(rng.normal(0.0, 1.0, xz.size))
                mw = measure(zs(xz + zs(rw)))
                if mw is not None:
                    add(state, L, r.win_start, "rw", mw)
        # ---- stationary baseline: fit on the longest windows this state allows,
        #      simulate, cut into every shorter length
        for Lfit in sorted(LENGTHS, reverse=True):
            win = make_windows(iv[state], state, window=float(Lfit), step=float(Lfit))
            if len(win) >= 3:
                break
        else:
            continue
        for r in win.itertuples(index=False):
            seg = slice_sorted(sp, r.win_start, r.win_stop)
            x = binned_counts(seg, r.win_start, r.win_stop, BIN, presliced=True)
            if x.size < P + 20 or x.std() == 0:
                continue
            fit = fit_ar(zs(x))
            if fit is None:
                continue
            sim = simulate_ar(fit[0], fit[1], x.size, rng)
            if sim is None:
                continue
            for L in LENGTHS:
                if L > Lfit:
                    continue
                nL = int(round(L / BIN))
                for k in range(sim.size // nL):
                    piece = sim[k * nL:(k + 1) * nL]
                    if piece.std() == 0:
                        continue
                    ms = measure(zs(piece))
                    if ms is not None:
                        add(state, L, r.win_start + k * L, "stationary", ms, dict(Lfit=Lfit))
    return animal, day, pd.DataFrame(rows), None


# ----------------------------------------------------------------------------- summary
def _sess(D):
    return D.assign(sess=D.animal.astype(str) + "_" + D.day.astype(str))


def summarise(D):
    D = _sess(D)
    rows = []
    for (state, L, v), g in D.groupby(["state", "L", "variant"]):
        if g.d2.notna().sum() < 10:
            continue
        per = g.groupby("sess").d2.median().dropna()
        lam = g.groupby("sess").apply(
            lambda h: 1 - np.nanmean(h.sd ** 2) / np.nanvar(h.d2, ddof=1)
            if h.d2.notna().sum() > 10 else np.nan)
        rows.append(dict(state=state, L=L, variant=v, n_windows=int(g.d2.notna().sum()),
                         n_sessions=len(per), d2_median=float(per.median()),
                         d2_q25=float(per.quantile(.25)), d2_q75=float(per.quantile(.75)),
                         sumphi_median=float(g.sumphi.median()),
                         adf_reject_frac=float((g.p_adf < .05).mean()),
                         kpss_pass_frac=float((g.p_kpss > .05).mean()),
                         lambda_median=float(np.nanmedian(lam))))
    return pd.DataFrame(rows)


def _stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def report(D, S):
    D = _sess(D)
    print("\n" + "=" * 118)
    print("A. d2 (2024 Euclidean, p=20) vs window length -- median over sessions of per-session medians")
    print("=" * 118)
    for state in STATES:
        s = S[S.state == state]
        if not len(s):
            continue
        print(f"\n  {state}  (n sessions at 30 s: {int(s[(s.L == 30) & (s.variant == 'real')].n_sessions.sum())})")
        print(f"  {'L (s)':>6}" + "".join(f"{v:>11s}" for v in VARIANTS)
              + f" | {'ADF rej':>8}{'KPSS ok':>8}  (real) | {'ADF rej':>8}{'KPSS ok':>8} (stationary) | {'ADF rej':>8} (rw)"
              + f" | {'lambda':>6}")
        for L in LENGTHS:
            r = s[s.L == L].set_index("variant")
            if "real" not in r.index:
                continue
            g = lambda v, c: r[c].get(v, np.nan)                  # noqa: E731
            print(f"  {L:>6}" + "".join(f"{g(v, 'd2_median'):>11.4f}" for v in VARIANTS)
                  + f" | {g('real', 'adf_reject_frac'):>8.0%}{g('real', 'kpss_pass_frac'):>8.0%}         "
                  + f"| {g('stationary', 'adf_reject_frac'):>8.0%}{g('stationary', 'kpss_pass_frac'):>8.0%}              "
                  + f"| {g('rw', 'adf_reject_frac'):>8.0%}      | {g('real', 'lambda_median'):>6.2f}")

    print("\n" + "=" * 118)
    print("A'. per-session slope of d2 vs log2(window length) [d2 per octave]; signed-rank across sessions")
    print("    'real - stationary' is the test: ~0 means the data fall with length only as a stationary AR(20) does")
    print("=" * 118)
    for state in STATES:
        g = D[D.state == state]
        if not len(g):
            continue
        per = g.groupby(["sess", "L", "variant"]).d2.median().unstack("variant").reset_index()
        line = f"  {state:9s}"
        for v in ("real", "detrended", "stationary", "ramp1", "rw", "real-stationary"):
            slopes = []
            for sess, h in per.groupby("sess"):
                h = h.sort_values("L")
                y = (h["real"] - h["stationary"]).to_numpy() if v == "real-stationary" else h.get(v, pd.Series(np.nan, index=h.index)).to_numpy()
                ok = np.isfinite(y)
                if ok.sum() < 4:
                    continue
                slopes.append(stats.linregress(np.log2(h["L"].to_numpy()[ok]), y[ok]).slope)
            slopes = np.asarray(slopes)
            if slopes.size < 3:
                line += f" | {v}: n/a"
                continue
            pw = stats.wilcoxon(slopes).pvalue
            line += f" | {v}: {np.median(slopes):+.4f} ({int((slopes < 0).sum())}/{slopes.size} neg, p={pw:.3f})"
        print(line)

    print("\n" + "=" * 118)
    print("B. detrended vs real at 30 s: per-session median difference, signed-rank")
    print("=" * 118)
    g30 = D[D.L == 30]
    for state in STATES:
        g = g30[g30.state == state]
        per = g.groupby(["sess", "variant"]).d2.median().unstack("variant")
        if "detrended" not in per or per["detrended"].notna().sum() < 3:
            continue
        d = (per["detrended"] - per["real"]).dropna()
        print(f"  {state:9s} detrended - real: median {d.median():+.4f} ({(d > 0).sum()}/{len(d)} pos, "
              f"p={stats.wilcoxon(d).pvalue:.3f}); relative {100 * d.median() / per['real'].median():+.1f}%")

    print("\n" + "=" * 118)
    print("C. d2 vs the drift actually present in the window (30 s, real): within-session Spearman r vs |trend|")
    print("=" * 118)
    R = g30[g30.variant == "real"]
    for state in STATES:
        g = R[R.state == state]
        rr = np.asarray([stats.spearmanr(h.trend_sd.abs(), h.d2).correlation
                         for _, h in g.groupby("sess") if h.d2.notna().sum() > 20])
        rr = rr[np.isfinite(rr)]
        if rr.size < 3:
            print(f"  {state:9s} n/a (sessions with >20 windows: {rr.size})")
            continue
        print(f"  {state:9s} r = {np.median(rr):+.3f} ({int((rr < 0).sum())}/{rr.size} neg, p={stats.wilcoxon(rr).pvalue:.3f}); "
              f"observed |trend| median {g.trend_sd.abs().median():.2f} SD, p95 {g.trend_sd.abs().quantile(.95):.2f} SD "
              f"[ramp surrogates add 1, 2, 4 SD end-to-end]")
    # state contrasts on the lowest-drift tercile
    print("\n  State contrasts (paired across sessions, 30 s) -- all windows vs the lowest-drift tercile of each session x state:")
    R = R.copy()
    R["low_drift"] = R.groupby(["sess", "state"]).trend_sd.transform(
        lambda v: v.abs() <= v.abs().quantile(1 / 3))
    for a, b in (("NREM", "nonTHETA"), ("NREM", "THETA"), ("NREM", "REM")):
        for lab, sub in (("all", R), ("low-drift", R[R.low_drift])):
            per = sub.groupby(["sess", "state"]).d2.median().unstack("state")
            if a not in per or b not in per:
                continue
            d = (per[a] - per[b]).dropna()
            if len(d) < 3:
                continue
            print(f"    {a} - {b:8s} [{lab:9s}] median {d.median():+.4f}, {int((d < 0).sum())}/{len(d)} neg, "
                  f"p = {stats.wilcoxon(d).pvalue:.4f} {_stars(stats.wilcoxon(d).pvalue)}")


# ----------------------------------------------------------------------------- figure
def plot(D, S):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 10, "axes.linewidth": 1.2})
    D = _sess(D)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    ax = axes[0]
    for state in STATES:
        s = S[S.state == state]
        for v, ls, lw in (("real", "-", 2.2), ("stationary", "--", 1.4)):
            r = s[s.variant == v].sort_values("L")
            if not len(r):
                continue
            ax.plot(r.L, r.d2_median, ls, color=STATE_COLOR[state], lw=lw,
                    label=f"{state}" if v == "real" else None)
            if v == "real":
                ax.fill_between(r.L, r.d2_q25, r.d2_q75, color=STATE_COLOR[state], alpha=0.12, lw=0)
    for v, ls, lab in (("ramp1", ":", "NREM + linear drift (±1 SD)"), ("rw", "-.", "NREM + random walk (1 SD)")):
        r = S[(S.state == "NREM") & (S.variant == v)].sort_values("L")
        if len(r):
            ax.plot(r.L, r.d2_median, ls, color="0.35", lw=1.4, label=lab)
    ax.plot([], [], "-", color="k", lw=2.2, label="data")
    ax.plot([], [], "--", color="k", lw=1.4, label="stationary surrogate (AR(20) fitted at 120 s)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENGTHS)
    ax.set_xticklabels([str(L) for L in LENGTHS])
    ax.axvspan(9, 15.5, color="0.92", zorder=0)
    ax.text(12, 0.005, "N/p < 20", fontsize=7.5, ha="center", color="0.4")
    ax.set_ylim(bottom=0)
    ax.set_xlabel("window length (s)")
    ax.set_ylabel("d2 (2024 Euclidean, p = 20)")
    ax.set_title("A  d2 vs window length", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7, ncol=2)

    ax = axes[1]
    xs = np.arange(len(STATES))
    wv = [("real", "data", 1.0), ("stationary", "stationary surrogate", 0.55),
          ("ramp1", "+ linear drift", 0.3), ("rw", "+ random walk", 0.3)]
    wd = 0.2
    for j, (v, lab, al) in enumerate(wv):
        vals = []
        for st_ in STATES:
            q = S[(S.state == st_) & (S.L == 30) & (S.variant == v)]
            vals.append(float(q.adf_reject_frac.iloc[0]) if len(q) else np.nan)
        ax.bar(xs + (j - 1.5) * wd, vals, wd, color=[STATE_COLOR[st_] for st_ in STATES],
               alpha=al, edgecolor="k", lw=0.6, label=lab)
    ax.set_xticks(xs)
    ax.set_xticklabels(STATES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction of 30 s windows rejecting a unit root\n(ADF, p < 0.05)")
    ax.set_title("B  within-window unit-root test (30 s)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="lower left")

    ax = axes[2]
    g = D[(D.L == 30) & (D.variant == "real")]
    for state in STATES:
        h = g[g.state == state]
        if len(h):
            ax.plot(h.trend_sd.abs(), h.d2, ".", ms=2, alpha=0.2, color=STATE_COLOR[state], label=state, rasterized=True)
    ax.set_xlabel("|linear trend within window| (SD, end-to-end)")
    ax.set_ylabel("d2 (2024 Euclidean, p = 20), 30 s")
    ax.set_title("C  d2 vs the drift present in each window", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=7.5, markerscale=5)
    for a in axes:
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    assert_local_path(FIGDIR)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"r25_stationarity.{ext}")), dpi=300)
    print(f"\nfigure -> {FIGDIR}\\r25_stationarity.png/.svg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="PPP20 day4 only")
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    win_csv = os.path.join(LOGDIR, "r25_stationarity_windows.csv")
    sum_csv = os.path.join(LOGDIR, "r25_stationarity_summary.csv")
    if args.report_only:
        D = pd.read_csv(win_csv)
        S = summarise(D)
        report(D, S)
        plot(D, S)
        return
    sessions = [s for s in FIG1_SESSIONS if s.endswith(r"PPP20\day4")] if args.test else FIG1_SESSIONS
    t0 = time.time()
    print(f"R2.5 battery | lengths {LENGTHS} s | p={P} | {len(sessions)} sessions | {args.workers} workers", flush=True)
    out = []
    if args.test:
        for bp in sessions:
            a, d, df, err = one_session(bp)
            print(f"  {a}_{d}: {err or f'{len(df)} rows'}", flush=True)
            if df is not None:
                out.append(df)
    else:
        from multiprocessing import Pool
        with Pool(processes=args.workers) as pool:
            for a, d, df, err in pool.imap_unordered(one_session, sessions):
                print(f"  {a}_{d:9s} {err or f'{len(df):7d} rows'}", flush=True)
                if df is not None:
                    out.append(df)
    D = pd.concat(out, ignore_index=True)
    assert_local_path(LOGDIR)
    if not args.test:
        D.to_csv(assert_local_path(win_csv), index=False)
    S = summarise(D)
    if not args.test:
        S.to_csv(assert_local_path(sum_csv), index=False)
    print(f"\n{len(D)} rows in {time.time() - t0:.0f} s")
    report(D, S)
    if not args.test:
        plot(D, S)


if __name__ == "__main__":
    main()
