"""
r16_event_dissociation.py -- burst metrics and d2 against SWR rate and barrage rate (the Figure 2/3
double dissociation).

Given metrics like E/I and MUA that already track burst activity, d2's advantage over them is
qualitative rather than a gain in variance explained for SWR rate alone: within the same NREM
windows, d2 rises with ripple rate and falls with barrage rate, while rate/burstiness metrics
rise with ripple rate and stay flat on barrage rate.

  A. Fig-1 cohort (fingerprint table, 16 sessions, NREM non-overlapping windows):
     within-session Spearman r of each variable with ripple rate; partial r given the
     others; Shapley share of held-out linear R^2 (block CV) among
     {d2, MUA, Fano, E/I, synchrony}.
  B. Fig-3 barrage cohort (12 sessions, the windows of r25_drift_control_rates.py):
     per window MUA, Fano, MUA_std and population synchrony (SD across 50 ms bins of
     the fraction of units active) from the CA1 spikes; within-session Spearman r of
     each with ripple rate and with barrage rate; sessions with opposite signs; partial
     r of d2 with each rate given MUA, Fano and synchrony.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/r16_event_dissociation.csv              (per cohort x metric x session)
    -> data/revision/log/barrage_burst_metrics_windows.csv       (part B window table)
    -> reports/revision_figs/r16_event_dissociation.{png,svg}
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings
from math import comb

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                                  # noqa: E402
from fig1_metrics import MIN_FIRING_RATE, binned_counts, slice_sorted           # noqa: E402
from barrage_order_kernel_sweep import BARRAGE_SESSIONS                         # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
FINGERPRINT = os.path.join(LOGDIR, "fig1_state_space_kl_join.csv")
DRIFT_WIN = os.path.join(LOGDIR, "r25_drift_control_rates_windows.csv")
OUT_WIN = os.path.join(LOGDIR, "barrage_burst_metrics_windows.csv")
OUT_CSV = os.path.join(LOGDIR, "r16_event_dissociation.csv")
WIN_S = 30.0
BIN = 0.040
SYNC_BIN = 0.050
MIN_WIN = 20


def partial_spearman(a, b, C):
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    A = np.column_stack([np.ones(len(ra))] + [stats.rankdata(c) for c in C.T])
    ea = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
    eb = rb - A @ np.linalg.lstsq(A, rb, rcond=None)[0]
    return float(stats.pearsonr(ea, eb)[0])


def cv_r2(Xm, y, n_fold=5):
    n = len(y)
    b = np.linspace(0, n, n_fold + 1).astype(int)
    pred = np.full(n, np.nan)
    for f in range(n_fold):
        te = np.zeros(n, bool)
        te[b[f]:b[f + 1]] = True
        tr = ~te
        A = np.column_stack([np.ones(tr.sum()), Xm[tr]])
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred[te] = np.column_stack([np.ones(te.sum()), Xm[te]]) @ beta
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def shapley_shares(Xa, y, names):
    p = len(names)
    gains = {(): 0.0}
    for r in range(1, p + 1):
        for S in itertools.combinations(range(p), r):
            gains[S] = cv_r2(Xa[:, list(S)], y)
    full = gains[tuple(range(p))]
    phi = np.zeros(p)
    for j in range(p):
        for r in range(p):
            for S in itertools.combinations([i for i in range(p) if i != j], r):
                phi[j] += (gains[tuple(sorted(S + (j,)))] - gains[tuple(sorted(S))]) / (p * comb(p - 1, r))
    return full, phi


# ------------------------------------------------------------------ A: Fig-1 cohort, ripple rate
def part_a(rows):
    D = pd.read_csv(FINGERPRINT)
    N = D[D.state == "NREM"].copy()
    X = [("d2", "d2 (Euclidean)"), ("d2_kl", "d2 (KL)"), ("MUA_mean", "MUA"), ("MUA_fano", "MUA Fano"),
         ("EI_logratio", "E/I"), ("synchrony_sub81_mean", "synchrony"), ("isi_entropy_pooled", "ISI entropy"),
         ("BR_pyr_k200", "branching ratio")]
    covs = ["MUA_mean", "MUA_fano", "EI_logratio", "synchrony_sub81_mean", "isi_entropy_pooled"]
    print("\nA. Fig-1 cohort, NREM, 16 sessions: who predicts ripple rate?")
    print(f"  {'variable':16s}{'r(x, ripple)':>14s}{'pos':>7s}{'p':>8s} | {'partial | others':>17s}{'pos':>7s}{'p':>8s}")
    for c, lab in X:
        rs, ps = [], []
        for s, g in N.groupby("session"):
            h = g.dropna(subset=[c, "ripple_rate"])
            if len(h) <= MIN_WIN:
                continue
            rs.append(stats.spearmanr(h[c], h.ripple_rate).correlation)
            others = [o for o in covs if o != c] + ([] if c in ("d2", "d2_kl") else ["d2"])
            h2 = g.dropna(subset=[c, "ripple_rate"] + others)
            if len(h2) > 30:
                ps.append(partial_spearman(h2[c].to_numpy(), h2.ripple_rate.to_numpy(), h2[others].to_numpy(float)))
            rows.append(dict(cohort="fig1", metric=c, session=s, r_ripple=rs[-1],
                             partial_ripple=ps[-1] if len(h2) > 30 else np.nan))
        rs, ps = np.array(rs), np.array(ps)
        print(f"  {lab:16s}{np.median(rs):+14.3f}{int((rs > 0).sum()):>4d}/{rs.size:<2d}{stats.wilcoxon(rs).pvalue:8.4f} | "
              f"{np.median(ps):+17.3f}{int((ps > 0).sum()):>4d}/{ps.size:<2d}{stats.wilcoxon(ps).pvalue:8.4f}")
    preds = ["d2", "MUA_mean", "MUA_fano", "EI_logratio", "synchrony_sub81_mean"]
    shares = {p: [] for p in preds}
    fulls = []
    for s, g in N.groupby("session"):
        g = g.dropna(subset=preds + ["ripple_rate"])
        if len(g) < 60:
            continue
        y = np.log1p(g.ripple_rate.to_numpy())
        Xa = g[preds].to_numpy(float)
        Xa = (Xa - Xa.mean(0)) / Xa.std(0)
        full, phi = shapley_shares(Xa, y, preds)
        fulls.append(full)
        for j, p in enumerate(preds):
            shares[p].append(phi[j] / full if full > 0 else np.nan)
    print(f"  Shapley share of held-out R^2 (block CV; full model median R^2 {np.median(fulls):.2f}): "
          + ", ".join(f"{p.split('_')[0]} {np.nanmedian(shares[p]):+.0%}" for p in preds))


# ------------------------------------------------------------------ B: barrage cohort, both rates
def part_b(rows):
    from neuro_py.io import loading
    W = pd.read_csv(DRIFT_WIN)
    W = W[W.cohort == "barrage"]
    out = []
    for sess, bp in BARRAGE_SESSIONS.items():
        g = W[W.session == sess]
        if not len(g):
            continue
        st, cm = loading.load_spikes(bp, brainRegion="CA1")
        keep = np.asarray(cm.firingRate > MIN_FIRING_RATE)
        units = [np.sort(np.asarray(u, float)) for u in np.asarray(st.data, dtype=object)[keep]]
        sp = np.sort(np.concatenate(units))
        ctype = np.asarray(cm.putativeCellType.astype(str))[keep]
        e_sp = np.sort(np.concatenate([u for u, t in zip(units, ctype) if t == "Pyramidal Cell"] or [np.zeros(0)]))
        i_sp = np.sort(np.concatenate([u for u, t in zip(units, ctype) if "Interneuron" in t] or [np.zeros(0)]))
        n_e = int((ctype == "Pyramidal Cell").sum())
        n_i = int(sum("Interneuron" in t for t in ctype))
        for r in g.itertuples(index=False):
            w0, w1 = r.w, r.w + WIN_S
            x = binned_counts(slice_sorted(sp, w0, w1), w0, w1, BIN, presliced=True)
            act = np.array([binned_counts(slice_sorted(u, w0, w1), w0, w1, SYNC_BIN, presliced=True) > 0 for u in units])
            ne = slice_sorted(e_sp, w0, w1).size
            ni = slice_sorted(i_sp, w0, w1).size
            # E/I log-ratio: whole-window E and I rates
            # (per unit, so cell-count differences between sessions do not enter)
            ei = np.log((ne / max(n_e, 1) + 1e-3) / (ni / max(n_i, 1) + 1e-3)) if n_i > 0 else np.nan
            out.append(dict(session=sess, w=r.w, ripple_rate=r.ripple_rate, barrage_rate=r.barrage_rate, d2=r.d2_real,
                            MUA=x.mean() / BIN, fano=x.var() / x.mean() if x.mean() > 0 else np.nan,
                            MUA_std=x.std() / BIN, sync=act.mean(0).std(), EI=ei, n_e=n_e, n_i=n_i))
        print(f"  {sess:12s} {len(g):4d} windows, {len(units):3d} units ({n_e} E, {n_i} I)", flush=True)
    B = pd.DataFrame(out)
    assert_local_path(LOGDIR)
    B.to_csv(assert_local_path(OUT_WIN), index=False)
    print("\nB. barrage cohort, NREM, 12 sessions: r with ripple rate | r with barrage rate (median, sessions positive)")
    print(f"  {'variable':10s}{'ripple':>9s}{'pos':>7s}{'p':>8s} | {'barrage':>9s}{'pos':>7s}{'p':>8s} | {'opposite signs':>15s}")
    for c in ("d2", "MUA", "fano", "MUA_std", "sync", "EI"):
        rr, rb = [], []
        for s, h in B.groupby("session"):
            h = h.dropna(subset=[c, "ripple_rate", "barrage_rate"])
            if len(h) <= MIN_WIN:
                continue
            rr.append(stats.spearmanr(h[c], h.ripple_rate).correlation)
            rb.append(stats.spearmanr(h[c], h.barrage_rate).correlation)
            rows.append(dict(cohort="barrage", metric=c, session=s, r_ripple=rr[-1], r_barrage=rb[-1]))
        rr, rb = np.array(rr), np.array(rb)
        print(f"  {c:10s}{np.median(rr):+9.3f}{int((rr > 0).sum()):>4d}/{rr.size:<2d}{stats.wilcoxon(rr).pvalue:8.4f} | "
              f"{np.median(rb):+9.3f}{int((rb > 0).sum()):>4d}/{rb.size:<2d}{stats.wilcoxon(rb).pvalue:8.4f} | "
              f"{int(((rr > 0) != (rb > 0)).sum()):>8d}/{rr.size}")
    for tgt in ("barrage_rate", "ripple_rate"):
        ps = np.array([partial_spearman(h.d2.to_numpy(), h[tgt].to_numpy(), h[["MUA", "fano", "sync", "EI"]].to_numpy(float))
                       for _, h in B.dropna(subset=["EI"]).groupby("session") if len(h) > 30])
        print(f"  partial r(d2, {tgt:12s} | MUA, Fano, sync, E/I): {np.median(ps):+.3f} ({int((ps > 0).sum())}/{ps.size} pos, p={stats.wilcoxon(ps).pvalue:.4f})")
    return B


def plot(R):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 10, "axes.linewidth": 1.2})
    P = R[R.cohort == "barrage"]
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    style = {"d2": ("#7b3294", "o", "d2 (2024 Euclidean)"), "MUA": ("#fdb863", "s", "MUA"),
             "fano": ("#d95f0e", "^", "MUA Fano"), "MUA_std": ("#e08214", "v", "MUA std"),
             "sync": ("#8c510a", "D", "synchrony"), "EI": ("#1b9e77", "P", "E/I log-ratio")}
    for m, (col, mk, lab) in style.items():
        h = P[P.metric == m]
        ax.plot(h.r_ripple, h.r_barrage, mk, ms=6, color=col, mec="k", mew=0.4, alpha=0.85, label=lab)
        ax.plot(h.r_ripple.median(), h.r_barrage.median(), mk, ms=13, color=col, mec="k", mew=1.5, zorder=5)
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.axvline(0, ls="--", lw=1, color="k")
    ax.set_xlabel("Spearman r with SWR rate (per session)")
    ax.set_ylabel("Spearman r with barrage rate (per session)")
    ax.set_title("Barrage cohort, NREM: burst metrics see events,\nd2 tells the two event types apart", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    assert_local_path(FIGDIR)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"r16_event_dissociation.{ext}")), dpi=300)
    print(f"\nfigure -> {FIGDIR}\\r16_event_dissociation.png/.svg")


def main():
    rows = []
    part_a(rows)
    part_b(rows)
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(OUT_CSV), index=False)
    plot(R)


if __name__ == "__main__":
    main()
