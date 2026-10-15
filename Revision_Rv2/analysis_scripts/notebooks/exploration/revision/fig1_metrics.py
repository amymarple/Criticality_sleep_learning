"""
fig1_metrics.py -- shared window, state, and metric library for the Figure 1 analyses.

Builds state-pure interval algebra (NREM / REM / THETA / nonTHETA, each guarded at interval
ends) and sliding 30 s windows over them, flagging the exact non-overlapping subset so every
window-level metric can join to d2 on `center_s`. Provides branching-ratio estimation (r_k via
FFT autocovariance, then a curve_fit-free m fit by variable projection + grid scan, with and
without a constant offset, and a k_max sweep), ISI entropy, population synchrony (raw and
unit-subsampled), binned multi-unit counts, and a d2 (2024 Euclidean) timeline loader/join.
Imported by run_fig1_metrics.py and by other scripts in this folder for GUARD_S, STATES,
build_state_intervals, binned_counts, make_windows, and slice_sorted.

Run through the conda env that has nelpy/neuro_py:
    C:/Users/Cornell/.conda/envs/criticality/python.exe run_fig1_metrics.py --test

Reads raw sessions from the read-only network shares (via run_fig1_metrics.py); writes only to
data/revision/ on D:.
"""
from __future__ import annotations

import os
import pickle
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import scipy.io
import scipy.stats

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
WINDOW_S = 30.0        # shared by every window-level metric
STEP_S = 2.0           # sliding; statistics use the `nonoverlap` subset
GUARD_S = 5.0          # shaved off each scored interval end

BIN_MUA_S = 0.040      # identical to the d2 input series
BIN_BR_S = 0.004       # k_max*bin = 800 ms, matching d2's p*dt
BIN_SYNC_S = 0.025     # aggregated in pairs -> 50 ms
SYNC_AGG = 2

# 200 = the d2-kernel-matched choice (200*4ms = 796 ms vs d2's p*dt = 800 ms).
# 2500 = Wilting & Priesemann's own rat k_max (10 s), the fit range at which the
# state-level m passes their H_offset/H_tau consistency tests; a 30 s window
# (7500 bins) reaches this lag.
#
# Do NOT thin with --max-windows to save time -- it is proportional, so it deletes
# THETA and REM. Adjust STATE_WINDOW_CAP instead.
K_MAX_SWEEP = (25, 50, 100, 200, 500, 1000, 2500)
K_MAX_REF = 2500

ISI_N_BINS = 20        # log10(ISI) histogram; ceiling log2(20) = 4.32 bits
MIN_FIRING_RATE = 0.2  # Hz — matches the existing BR notebook
SYNC_SUBSAMPLE_DRAWS = 20

STATES = ("THETA", "nonTHETA", "REM", "NREM")


# ----------------------------------------------------------------------------
# Interval algebra
# ----------------------------------------------------------------------------
def shrink_intervals(iv, guard=GUARD_S):
    iv = np.asarray(iv, float)
    if iv.size == 0:
        return np.empty((0, 2), float)
    iv = np.c_[iv[:, 0] + guard, iv[:, 1] - guard]
    return iv[iv[:, 1] > iv[:, 0]]


def intersect_intervals(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return np.empty((0, 2), float)
    a = a[np.argsort(a[:, 0])]; b = b[np.argsort(b[:, 0])]
    out, ia, ib = [], 0, 0
    while ia < len(a) and ib < len(b):
        s = max(a[ia, 0], b[ib, 0]); e = min(a[ia, 1], b[ib, 1])
        if e > s:
            out.append([s, e])
        if a[ia, 1] < b[ib, 1]:
            ia += 1
        else:
            ib += 1
    return np.array(out, float) if out else np.empty((0, 2), float)


def subtract_intervals(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if a.size == 0:
        return np.empty((0, 2), float)
    if b.size == 0:
        return a.copy()
    out = []
    for s, e in a:
        cur = s
        for s2, e2 in b:
            if e2 <= cur or s2 >= e:
                continue
            if s2 > cur:
                out.append([cur, min(s2, e)])
            cur = max(cur, e2)
            if cur >= e:
                break
        if cur < e:
            out.append([cur, e])
    return np.array(out, float) if out else np.empty((0, 2), float)


def build_state_intervals(state_dict, guard=GUARD_S):
    """NREM / REM / THETA (wake+theta) / nonTHETA (wake-theta), all guarded."""
    def g(key):
        v = state_dict.get(key)
        return shrink_intervals(v, guard) if v is not None and len(v) else np.empty((0, 2), float)

    nrem, wake, rem, theta = g("NREMstate"), g("WAKEstate"), g("REMstate"), g("THETA")
    return {
        "NREM": nrem,
        "REM": rem,
        "THETA": intersect_intervals(wake, theta),
        "nonTHETA": subtract_intervals(wake, theta),
    }


# ----------------------------------------------------------------------------
# Windows — sliding, with an exact non-overlapping subset flagged per interval
# ----------------------------------------------------------------------------
def make_windows(intervals, state, window=WINDOW_S, step=STEP_S):
    """
    Sliding windows fully inside each interval.

    `nonoverlap` marks every (window/step)-th window *within an interval*, which
    tiles contiguously and is therefore identical to computing non-overlapping
    windows from scratch. Indexing must be per-interval, not global.
    """
    stride = int(round(window / step))
    rows = []
    for iv_idx, (s, e) in enumerate(np.asarray(intervals, float)):
        n = int(np.floor((e - s - window) / step)) + 1
        for i in range(max(n, 0)):
            ws = s + i * step
            we = ws + window
            if we > e + 1e-9:
                break
            rows.append({
                "state": state, "iv_idx": iv_idx, "w_idx": i,
                "win_start": ws, "win_stop": we, "center_s": 0.5 * (ws + we),
                "nonoverlap": (i % stride == 0),
            })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Branching ratio — r_k computed once, exponential re-fitted per k_max
# ----------------------------------------------------------------------------
def _mr_func(x, a, b):
    return np.abs(a) * np.abs(b) ** x


# Candidate m values, dense where it matters: geometric in the decay constant
# tau = -1/ln(m) (bins), so resolution is ~1% in tau everywhere, plus a small
# supercritical margin. 1.05**199 stays finite, so no overflow guard is needed.
_M_GRID = np.unique(np.concatenate([
    np.exp(-1.0 / np.logspace(np.log10(0.2), np.log10(2.0e5), 1500)),
    np.linspace(1.0 + 1e-9, 1.05, 60),
]))


def _wls_sse(m, kk, rr, w, sum_wrr):
    """Weighted SSE of r_k = b*m^k with the linear amplitude b eliminated.

    For fixed m the optimal b is (sum w x r)/(sum w x^2) with x = m^k, so
    SSE(m) = sum w r^2 - num^2/den. Vectorised over m.
    """
    m = np.atleast_1d(np.asarray(m, float))
    X = np.exp(np.outer(np.log(m), kk))          # (n_m, n_k)
    num = X @ (w * rr)
    den = (X * X) @ w
    good = den > 0
    return np.where(good, sum_wrr - num * num / np.where(good, den, 1.0), np.inf)


def _fit_exp_wls(kk, rr, w):
    """Global weighted least-squares fit of r_k = b*m^k. Returns (m, b).

    Variable projection reduces this to a smooth 1-D problem in m, solved by a
    grid scan plus golden-section refinement, which guarantees a global optimum
    rather than converging to a local minimum from a single start.
    """
    sum_wrr = float(np.dot(w, rr * rr))
    sse = _wls_sse(_M_GRID, kk, rr, w, sum_wrr)
    i = int(np.argmin(sse))
    if not np.isfinite(sse[i]):
        return np.nan, np.nan

    lo = _M_GRID[max(i - 1, 0)]
    hi = _M_GRID[min(i + 1, len(_M_GRID) - 1)]
    invphi, invphi2 = 0.6180339887498949, 0.3819660112501051
    c, d = lo + invphi2 * (hi - lo), lo + invphi * (hi - lo)
    fc = float(_wls_sse(c, kk, rr, w, sum_wrr)[0])
    fd = float(_wls_sse(d, kk, rr, w, sum_wrr)[0])
    for _ in range(60):
        if hi - lo < 1e-12:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = lo + invphi2 * (hi - lo)
            fc = float(_wls_sse(c, kk, rr, w, sum_wrr)[0])
        else:
            lo, c, fc = c, d, fd
            d = lo + invphi * (hi - lo)
            fd = float(_wls_sse(d, kk, rr, w, sum_wrr)[0])
    m = 0.5 * (lo + hi)

    x = np.exp(kk * np.log(m))
    den = float(np.dot(w, x * x))
    b = float(np.dot(w * x, rr) / den) if den > 0 else np.nan
    return m, b


def active_unit_counts(unit_spikes, t0, t1, bin_s=BIN_BR_S):
    """A_t = number of *units active* per bin, not total spikes.

    This is the input `generate_branching_ratio_mouse.ipynb` feeds to MR_estimation
    (`A_t = np.count_nonzero(spike_matrix, axis=0)`).
    """
    n = max(int(round((t1 - t0) / bin_s)), 1)
    A = np.zeros(n, np.int64)
    for u in unit_spikes:
        s = slice_sorted(u, t0, t1)
        if s.size:
            idx = ((s - t0) / bin_s).astype(np.int64)
            np.clip(idx, 0, n - 1, out=idx)
            A[np.unique(idx)] += 1          # each unit counts once per bin
    return A.astype(float)


def rk_curve(activity, k_max=K_MAX_REF):
    """
    Linear-regression slopes r_k for k = 1..k_max-1 (Wilting & Priesemann eq. 2).

    Exact, but O(N log N) instead of O(N*k_max): the cross term
    sum_i A[i]*A[i+k] is the raw autocorrelation, obtained for all lags at once
    by FFT; every other term comes from prefix sums. Identical to looping
    scipy.stats.linregress over lags.
    """
    A = np.asarray(activity, float)
    N = A.size
    k = np.arange(1, k_max)
    r_k = np.full(len(k), np.nan)
    se = np.full(len(k), np.nan)
    if N < 20:
        return k, r_k, se

    # prefix sums for the head (x = A[:N-k]) and tail (y = A[k:]) segments
    c1 = np.concatenate(([0.0], np.cumsum(A)))
    c2 = np.concatenate(([0.0], np.cumsum(A * A)))
    tot1, tot2 = c1[-1], c2[-1]

    # full raw autocorrelation via FFT: acf[k] = sum_i A[i]*A[i+k]
    nfft = 1 << int(np.ceil(np.log2(2 * N)))
    F = np.fft.rfft(A, n=nfft)
    acf = np.fft.irfft(F * np.conjugate(F), n=nfft)[:k_max]

    for i, lag in enumerate(k):
        n = N - lag
        if n < 10:
            break
        sx = c1[n]                 # sum A[0:n]
        sxx = c2[n]
        sy = tot1 - c1[lag]        # sum A[lag:N]
        syy = tot2 - c2[lag]
        sxy = acf[lag]
        Sxx = sxx - sx * sx / n
        if Sxx <= 0:
            continue
        Sxy = sxy - sx * sy / n
        Syy = syy - sy * sy / n
        slope = Sxy / Sxx
        resid = (Syy - slope * Sxy) / (n - 2)
        r_k[i] = slope
        se[i] = np.sqrt(max(resid, 0.0) / Sxx)
    return k, r_k, se


def fit_m(k, r_k, se, k_max_fit):
    """
    Fit r_k = b*m^k over k < k_max_fit, replicating WiltingPriesemann.MR_estimation
    (same p0, same sigma weighting, which is relative to the number of lags used).
    """
    mask = (k < k_max_fit) & np.isfinite(r_k) & np.isfinite(se)
    kk, rr, ss = k[mask].astype(float), r_k[mask], se[mask]
    if len(kk) < 5:
        return np.nan
    sigma = ss * np.linspace(1, 10, len(ss))
    bad = ~np.isfinite(sigma) | (sigma <= 0)
    if bad.all():
        sigma = np.ones_like(sigma)
    elif bad.any():
        sigma[bad] = np.nanmedian(sigma[~bad])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, b = _fit_exp_wls(kk, rr, 1.0 / (sigma * sigma))
        # b <= 0 means r_k does not decay from a positive amplitude: the
        # exponential model has not fitted anything and m is not interpretable.
        if not np.isfinite(m) or not np.isfinite(b) or b <= 0:
            return np.nan
        return float(m)
    except Exception:
        return np.nan


def _wls_sse_offset(m, kk, rr, w):
    """Weighted SSE of r_k = b*m^k + c, with BOTH linear parameters eliminated.

    Same variable-projection trick as _wls_sse, but the design matrix is [x, 1]
    instead of [x], so the normal equations are 2x2 and still solvable in closed
    form for every candidate m at once.
    """
    m = np.atleast_1d(np.asarray(m, float))
    X = np.exp(np.outer(np.log(m), kk))                 # (n_m, n_k)
    sw = float(w.sum())
    Sxx = (X * X) @ w
    Sx = X @ w
    Sxr = X @ (w * rr)
    Sr = float(np.dot(w, rr))
    det = Sxx * sw - Sx * Sx
    ok = np.abs(det) > 1e-300
    b = np.where(ok, (Sxr * sw - Sx * Sr) / np.where(ok, det, 1.0), 0.0)
    c = np.where(ok, (Sxx * Sr - Sx * Sxr) / np.where(ok, det, 1.0), 0.0)
    sse = float(np.dot(w, rr * rr)) - (b * Sxr + c * Sr)
    return np.where(ok, sse, np.inf), b, c


def fit_m_offset(k, r_k, se, k_max_fit):
    """Fit r_k = b*m^k + c. Returns (m, b, c, sse_offset, sse_plain).

    W&P's primary model has no offset; their consistency check `H_offset`
    (Supplementary Note 5) REJECTS a recording when r_k carries one, because a
    constant offset is not explained by a branching process. A flat non-zero tail
    drags the plain fit's m toward 1 over long lag ranges, so this is the diagnostic
    that says whether a high m at large k_max is real decay or an artifact.
    """
    mask = (k < k_max_fit) & np.isfinite(r_k) & np.isfinite(se)
    kk, rr, ss = k[mask].astype(float), r_k[mask], se[mask]
    if len(kk) < 6:
        return (np.nan,) * 5
    sigma = ss * np.linspace(1, 10, len(ss))
    bad = ~np.isfinite(sigma) | (sigma <= 0)
    if bad.all():
        sigma = np.ones_like(sigma)
    elif bad.any():
        sigma[bad] = np.nanmedian(sigma[~bad])
    w = 1.0 / (sigma * sigma)

    sse, b_all, c_all = _wls_sse_offset(_M_GRID, kk, rr, w)
    i = int(np.argmin(sse))
    if not np.isfinite(sse[i]):
        return (np.nan,) * 5
    lo, hi = _M_GRID[max(i - 1, 0)], _M_GRID[min(i + 1, len(_M_GRID) - 1)]
    grid = np.linspace(lo, hi, 200)
    s2, b2, c2 = _wls_sse_offset(grid, kk, rr, w)
    j = int(np.argmin(s2))
    m_off = float(grid[j])

    m_plain, _ = _fit_exp_wls(kk, rr, w)
    sse_plain = float(_wls_sse(np.array([m_plain]), kk, rr, w,
                               float(np.dot(w, rr * rr)))[0])
    return m_off, float(b2[j]), float(c2[j]), float(s2[j]), sse_plain


def tau_ms(m, dt_s=BIN_BR_S):
    m = np.asarray(m, float)
    out = np.full(m.shape, np.nan)
    ok = np.isfinite(m) & (m > 0) & (m < 1)
    out[ok] = -dt_s * 1000.0 / np.log(m[ok])
    return out if out.ndim else float(out)


# ----------------------------------------------------------------------------
# Window-level metrics
# ----------------------------------------------------------------------------
def isi_entropy(spike_times, n_bins=ISI_N_BINS):
    """Shannon entropy (bits) of the log10(ISI) distribution. Scale-invariant: a
    rate change translates log10(ISI), the auto-ranged histogram translates with
    it, entropy is unchanged."""
    st = np.asarray(spike_times, float)
    st = np.unique(st[np.isfinite(st)])
    if st.size < 3:
        return np.nan
    isis = np.diff(st)
    isis = isis[np.isfinite(isis) & (isis > 0)]
    if isis.size < 2:
        return np.nan
    hist, _ = np.histogram(np.log10(isis), bins=n_bins)
    n = hist.sum()
    if n <= 0:
        return np.nan
    p = hist.astype(float) / n
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def slice_sorted(sorted_times, t0, t1):
    """O(log N) window slice of a pre-sorted spike array (never a boolean scan)."""
    lo, hi = np.searchsorted(sorted_times, (t0, t1))
    return sorted_times[lo:hi]


def binned_counts(times, t0, t1, bin_s, presliced=False):
    edges = np.arange(t0, t1 + bin_s, bin_s)
    if len(edges) < 2:
        return np.zeros(0)
    if not presliced:
        times = slice_sorted(times, t0, t1)
    return np.histogram(times, bins=edges)[0].astype(float)


def active_matrix(unit_spikes_sorted, t0, t1, bin_s=BIN_SYNC_S, agg=SYNC_AGG):
    """
    (n_units, n_chunks) boolean: did unit u fire in chunk c?

    Built once per window; both raw and subsampled synchrony are derived from it
    by summing rows, so bootstrap draws cost no extra histogramming.
    """
    n_units = len(unit_spikes_sorted)
    edges = np.arange(t0, t1 + bin_s, bin_s)
    nb = len(edges) - 1
    if n_units == 0 or nb < agg:
        return np.zeros((n_units, 0), dtype=bool)
    mat = np.zeros((n_units, nb), dtype=np.int32)
    for i, sp in enumerate(unit_spikes_sorted):
        s = slice_sorted(sp, t0, t1)
        if s.size:
            mat[i] = np.histogram(s, bins=edges)[0]
    n_chunk = nb // agg
    act = mat[:, :n_chunk * agg].reshape(n_units, n_chunk, agg).sum(axis=2)
    return act > 0


def synchrony_from_matrix(act):
    if act.shape[0] == 0 or act.shape[1] == 0:
        return np.nan, np.nan
    s = act.sum(axis=0) / float(act.shape[0])
    return float(s.mean()), float(s.std())


def synchrony_subsampled_from_matrix(act, n_target, rng, draws=SYNC_SUBSAMPLE_DRAWS):
    """Median synchrony over `draws` random unit subsets — row selection only."""
    n_units = act.shape[0]
    if n_units == 0 or act.shape[1] == 0 or not n_target or n_units < n_target:
        return np.nan, np.nan
    means = np.empty(draws)
    stds = np.empty(draws)
    for d in range(draws):
        idx = rng.choice(n_units, size=n_target, replace=False)
        s = act[idx].sum(axis=0) / float(n_target)
        means[d], stds[d] = s.mean(), s.std()
    return float(np.median(means)), float(np.median(stds))


# ----------------------------------------------------------------------------
# d2 join (optional — the extractor still produces everything without it)
# ----------------------------------------------------------------------------
D2_GLOBS = [
    "*_{region}_D2_time_resolved_error_bar_forplot.mat",
    "*_{region}_D2_time_resolved_error_bar.mat",
    "*_{region}_D2_time_resolved.mat",
    "*_{region}_D2_time.mat",
]


# The MATLAB d2 scores WAKE as one state; these windows split it into THETA and
# nonTHETA, so both map onto the same d2 series.
D2_STATE_MAP = {"THETA": "WAKE", "nonTHETA": "WAKE", "REM": "REM", "NREM": "NREM"}


def _as_dict(o):
    """mat_struct -> dict, dict -> itself."""
    if hasattr(o, "_fieldnames"):
        return {k: getattr(o, k) for k in o._fieldnames}
    return o if isinstance(o, dict) else None


def _d2_from_results(res):
    """`results` is a struct array with one entry per state: state/center/d2/d2_err."""
    frames = []
    for e in np.atleast_1d(np.asarray(res, dtype=object)).ravel():
        d = _as_dict(e)
        if not d or "center" not in d or "d2" not in d:
            continue
        c = np.asarray(d["center"], float).ravel()
        v = np.asarray(d["d2"], float).ravel()
        err = (np.asarray(d["d2_err"], float).ravel()
               if "d2_err" in d and np.size(d["d2_err"]) == c.size
               else np.full(c.size, np.nan))
        n = min(c.size, v.size)
        if not n:
            continue
        frames.append(pd.DataFrame({
            "center_s": c[:n], "d2": v[:n], "d2_err": err[:n],
            "d2_state": str(np.asarray(d.get("state", "")).ravel()[0]).strip().upper(),
        }))
    return pd.concat(frames, ignore_index=True) if frames else None


def _d2_from_timeline(tl):
    """Legacy flat layout: a single `timeline` struct with center_s/d2."""
    d = _as_dict(tl)
    if not d or "center_s" not in d or "d2" not in d:
        return None
    c = np.asarray(d["center_s"], float).ravel()
    return pd.DataFrame({
        "center_s": c,
        "d2": np.asarray(d["d2"], float).ravel(),
        "d2_err": (np.asarray(d["d2_err"], float).ravel()
                   if "d2_err" in d else np.full(c.size, np.nan)),
        "d2_state": "",
    })


def load_d2_timeline(basepath, region):
    """Return (DataFrame[center_s, d2, d2_err, d2_state], source) or (None, why)."""
    import glob as _glob
    sdir = os.path.join(basepath, "complex_system")
    if not os.path.isdir(sdir):
        return None, "no complex_system dir"
    hit = None
    for pat in D2_GLOBS:
        f = sorted(_glob.glob(os.path.join(sdir, pat.format(region=region))))
        if f:
            hit = f[-1]
            break
    if hit is None:
        return None, "no *_D2_time*.mat"

    md, why = None, ""
    try:
        md = scipy.io.loadmat(hit, squeeze_me=True, struct_as_record=False)
    except Exception:
        try:
            import mat73
            md = mat73.loadmat(hit)
        except Exception as e:
            why = "%s" % type(e).__name__
    if md is None:
        return None, "unreadable (%s): %s" % (os.path.basename(hit), why)

    for key, parse in (("results", _d2_from_results), ("timeline", _d2_from_timeline)):
        if key in md:
            df = parse(md[key])
            if df is not None and len(df):
                return df, os.path.basename(hit)
    return None, "no results/timeline in %s" % os.path.basename(hit)


def join_d2(win_df, d2_df, tol_s=STEP_S / 2.0):
    """Nearest-center join, matched within state so a NREM window can never pick
    up a WAKE d2 point."""
    if d2_df is None or not len(d2_df):
        win_df["d2"] = np.nan
        win_df["d2_err"] = np.nan
        return win_df, 0.0

    win_df = win_df.copy()
    win_df["d2_state"] = win_df["state"].map(D2_STATE_MAP).fillna("")
    by = ["d2_state"] if d2_df["d2_state"].astype(bool).any() else None
    if by is None:
        win_df["d2_state"] = ""

    out = pd.merge_asof(
        win_df.sort_values("center_s"),
        d2_df.sort_values("center_s")[["center_s", "d2", "d2_err", "d2_state"]],
        on="center_s", by=by, direction="nearest", tolerance=tol_s,
    )
    rate = float(np.mean(np.isfinite(out["d2"]))) if len(out) else 0.0
    return out, rate
