"""
r35_sham_excision.py -- sham-excision control: a location-matched sham arm removing the
same number and duration of intervals as the ripples, built alongside the full and
ripple-excised arms of d2 per window.

Builds three arms per 30 s NREM window from the 40 ms spike-count series: full (the window
intact), excised (bins overlapping a ripple interval dropped and the remainder glued, any
free segment shorter than MIN_FREE_S discarded; excision removes a median 7-9% of each
window), and sham (the same number of intervals and durations as the ripples, placed
uniformly at random where they overlap neither a ripple nor each other, removed and glued
the same way, averaged over N_SHAM draws per window). d2 (2024 Euclidean, AR 20) is
computed in Python on each arm (full corresponds to the published d2, excised to the
ripple-ablated d2). Each arm then goes through the
same per-SWR binomial ladder (d2 alone, + MUA, + MUA + MUA_std, + MUA + MUA_std + SWR
rate), session as the unit of analysis.

Inputs: fig2f_background_state_windows.csv (windows, covariates, replay counts); raw
spikes and ripple intervals from the session shares (SESSIONS, r35_swr_property_control.py).
Reads the shares read-only; writes only to D:.

    -> data/revision/log/r35_sham_excision_windows.csv
    -> data/revision/log/r35_sham_excision.csv
"""
from __future__ import annotations

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
from run_fig1_metrics import assert_local_path                      # noqa: E402
from fig1_metrics import MIN_FIRING_RATE                            # noqa: E402
from r35_swr_property_control import SESSIONS                       # noqa: E402
from r35_binomial_replay import fit_binomial, zs                    # noqa: E402
from r35_binomial_fig2fvars import LOGDIR, WIN_S                    # noqa: E402
from euclid_d2_err_orders import d2_and_sd                          # noqa: E402

BIN = 0.040
P = 20
MIN_FREE_S = 1.0
N_SHAM = 5
SEED = 20260917
WIN_CSV = os.path.join(LOGDIR, "fig2f_background_state_windows.csv")
OUT_WIN = os.path.join(LOGDIR, "r35_sham_excision_windows.csv")
OUT_CSV = os.path.join(LOGDIR, "r35_sham_excision.csv")
ARMS = ["d2_published", "d2_excised", "py_full", "py_excised", "py_sham"]
MODELS = [("M1", ["d2_z"], "d2"), ("M2", ["d2_z", "MUA_z"], "d2 + MUA"),
          ("M3", ["d2_z", "MUA_z", "MUA_std_z"], "d2 + MUA + MUA_std"),
          ("M5", ["d2_z", "MUA_z", "MUA_std_z", "rate_z"], "d2 + MUA + MUA_std + SWR rate")]


def d2_of(counts):
    if counts.size < P + 20 or counts.std() == 0:
        return np.nan
    o = d2_and_sd((counts - counts.mean()) / counts.std(), P)
    return np.nan if o is None else o[0]


def glue(counts, drop_mask):
    """Keep bins not in drop_mask, discarding kept runs shorter than MIN_FREE_S."""
    keep = ~drop_mask
    d = np.diff(np.r_[0, keep.astype(int), 0])
    st, en = np.where(d == 1)[0], np.where(d == -1)[0]
    min_bins = int(round(MIN_FREE_S / BIN))
    parts = [counts[a:b] for a, b in zip(st, en) if (b - a) >= min_bins]
    return np.concatenate(parts) if parts else np.zeros(0)


def bin_mask(edges, iv):
    """Bins overlapping any [start, stop) in iv."""
    m = np.zeros(len(edges) - 1, bool)
    if not len(iv):
        return m
    s, e = iv[:, 0], iv[:, 1]
    # vectorised: bin i overlaps some interval if any(s < edge_hi) & (e > edge_lo)
    lo, hi = edges[:-1], edges[1:]
    order = np.argsort(s)
    s, e = s[order], e[order]
    cummax_e = np.maximum.accumulate(e)
    j = np.searchsorted(s, hi, "left")          # intervals starting before bin end
    m = (j > 0) & (cummax_e[np.clip(j - 1, 0, len(e) - 1)] > lo)
    return m


def sham_intervals(rip_in, w0, w1, rng):
    """Same number and durations as rip_in, placed at random inside [w0, w1] with no
    overlap with rip_in or each other."""
    durs = rip_in[:, 1] - rip_in[:, 0]
    out = []
    occupied = list(map(tuple, rip_in))
    for du in durs:
        for _ in range(200):
            a = rng.uniform(w0, w1 - du)
            b = a + du
            if all(b <= s or a >= e for s, e in occupied):
                occupied.append((a, b))
                out.append((a, b))
                break
    return np.array(out) if out else np.zeros((0, 2))


def one_session(sess, W):
    from neuro_py.io import loading
    bp = SESSIONS[sess]
    st, cm = loading.load_spikes(bp, brainRegion="CA1")
    keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
    sp = np.sort(np.concatenate([np.asarray(u, float) for u in np.asarray(st.data, dtype=object)[keep]]))
    rip = loading.load_ripples_events(bp)
    riv = np.sort(np.array(rip[["start", "stop"]], float), axis=0)
    rng = np.random.default_rng(SEED + hash(sess) % 10_000)
    rows = []
    nb = int(round(WIN_S / BIN))
    for r in W.itertuples(index=False):
        w0 = r.center_s - WIN_S / 2
        edges = w0 + BIN * np.arange(nb + 1)
        lo, hi = np.searchsorted(sp, (edges[0], edges[-1]))
        c = np.histogram(sp[lo:hi], bins=edges)[0].astype(float)
        inw = riv[(riv[:, 1] > edges[0]) & (riv[:, 0] < edges[-1])]
        m_rip = bin_mask(edges, inw)
        py_full = d2_of(c)
        py_exc = d2_of(glue(c, m_rip))
        shams = []
        for _ in range(N_SHAM):
            siv = sham_intervals(inw, edges[0], edges[-1], rng)
            shams.append(d2_of(glue(c, bin_mask(edges, siv))))
        rows.append(dict(session=sess, center_s=r.center_s, py_full=py_full, py_excised=py_exc,
                         py_sham=float(np.nanmean(shams)) if len(shams) else np.nan,
                         frac_removed=float(m_rip.mean()), n_rip=len(inw)))
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    W = pd.read_csv(WIN_CSV)
    out = []
    for sess, g in W.groupby("session", sort=False):
        R = one_session(sess, g)
        m = g.merge(R, on=["session", "center_s"])
        r_full = m[["py_full", "d2_published"]].dropna().corr().iloc[0, 1]
        r_exc = m[["py_excised", "d2_excised"]].dropna().corr().iloc[0, 1]
        print(f"  {sess:12s} {len(m):4d} windows | validation r(py_full, published) {r_full:.3f}  "
              f"r(py_excised, .mat excised) {r_exc:.3f} | removed {m.frac_removed.median():.1%} | "
              f"median d2 full {m.py_full.median():.4f} excised {m.py_excised.median():.4f} sham {m.py_sham.median():.4f}", flush=True)
        out.append(m)
    D = pd.concat(out, ignore_index=True)
    for arm in ("py_full", "py_excised", "py_sham"):
        D[f"{arm}_z"] = D.groupby("session")[arm].transform(zs)
    assert_local_path(LOGDIR)
    D.to_csv(assert_local_path(OUT_WIN), index=False)
    rows = []
    for arm in ARMS:
        Da = D.copy()
        Da["d2_z"] = Da[f"{arm}_z"]
        for tag, cols, lbl in MODELS:
            for s, sub in Da.groupby("session", sort=False):
                r = fit_binomial(sub, cols)
                if r is not None:
                    rows.append(dict(arm=arm, model=tag, label=lbl, session=s, **r))
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(OUT_CSV), index=False)
    print(f"\n{len(D)} windows in {time.time() - t0:.0f} s")
    report(R, D)


def report(R, D):
    print("\n=== within-session correlations of the arms (median over sessions) ===")
    for a, b in (("py_full", "d2_published"), ("py_excised", "d2_excised"), ("py_sham", "py_full"),
                 ("py_sham", "py_excised"), ("py_excised", "py_full")):
        rs = [g[[a, b]].dropna().corr().iloc[0, 1] for _, g in D.groupby("session")]
        print(f"  r({a:11s}, {b:12s}) = {np.median(rs):.3f}")
    print("  r(d2, MUA) within session: " + "  ".join(
        f"{a} {np.median([g[[a, 'MUA']].dropna().corr().iloc[0, 1] for _, g in D.groupby('session')]):+.3f}"
        for a in ("py_full", "py_excised", "py_sham")))
    print("\n=== binomial ladder, b1 per SD of d2, session = unit ===")
    print(f"  {'arm':13s}" + "".join(f"{t:>28s}" for t, _, _ in MODELS))
    for arm in ARMS:
        line = f"  {arm:13s}"
        for tag, _, _ in MODELS:
            P_ = R[(R.arm == arm) & (R.model == tag)]
            b = P_.b1.to_numpy()
            pw = stats.wilcoxon(b).pvalue if b.size > 1 else np.nan
            line += f"{np.median(b):+8.4f} {int((b > 0).sum()):d}/{b.size} p={pw:.3f}     "
        print(line)


if __name__ == "__main__":
    main()
