"""
r38_hyperplane_explainer.py -- a worked, from-data introduction to what d2 (2024
Euclidean) is: the beta=2 hyperplane, a fitted AR(20) kernel, and one session's windows
shown as points relative to it (Sooter et al. 2025 Fig. 2f draws the same construction for
a toy AR(3); this uses real data and real windows).

Example session PPP20_day5, its non-overlapping 30 s windows
(`data/revision/PPP20_day5_CA1_fig1_metrics_nonoverlap.pkl`), NREM only. For every window
the AR(20) history kernel phi is refitted with the published recipe
(`d2_yw_matlab_match.getFR` + Yule-Walker on the z-scored 40 ms population count of all
CA1 units, as the MATLAB producer pools them), and |1 - sum(phi)| / sqrt(20) equals the
stored d2. Five panels: the raw data; the fitted
kernel; the fitted model's power spectrum against the window's own Welch spectrum and the
hyperplane's; the kernel geometry of every window in the session relative to the plane;
and a 1-D ruler of sum(phi) with d2 on a secondary axis.

    -> data/revision/log/r38_hyperplane_windows.csv
    -> reports/revision_figs/r38_hyperplane_explainer.png / .svg
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import FIG1_SESSIONS, assert_local_path, session_id   # noqa: E402
from fig1_metrics import MIN_FIRING_RATE                                    # noqa: E402
from d2_yw_matlab_match import getFR, DT, P                                 # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
EX = "PPP20_day5"
PKL = os.path.join(REPO, "data", "revision", f"{EX}_CA1_fig1_metrics_nonoverlap.pkl")
LOG = os.path.join(REPO, "data", "revision", "log")
OUT = os.path.join(REPO, "reports", "revision_figs", "r38_hyperplane_explainer")
WIN = 30.0
SHOW = ("NREM",)
RED = "#d62728"          # highlight color for the example window
COL = {"NREM": "#7b3294", "REM": "#d7b5d8", "THETA": "#d95f0e", "nonTHETA": "#fdb863"}
NAME = {"NREM": "NREM", "REM": "REM", "THETA": "wake, θ", "nonTHETA": "wake, non-θ"}
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})


def yw_kernel(x, p=P):
    """Yule-Walker AR(p) on one z-scored series, as myAutocorr + myYuleWalker (single segment)."""
    z = (x - x.mean()) / x.std(ddof=1)
    n = z.size
    r = np.array([np.dot(z[:n - j], z[j:]) / (n - j) for j in range(1, p + 1)])
    R = np.array([[1.0 if i == j else r[abs(i - j) - 1] for j in range(p)] for i in range(p)])
    phi = np.linalg.solve(R, r)
    return phi, float(1.0 - r @ phi)                   # kernel, innovation variance (z units)


def model_psd(phi, s2, f, dt=DT):
    """One-sided PSD per Hz of the AR model: 2 s2 dt / |1 - sum phi_k e^{-i 2 pi f k dt}|^2."""
    k = np.arange(1, phi.size + 1)
    h = 1 - np.exp(-2j * np.pi * np.outer(f, k) * dt) @ phi
    return 2 * s2 * dt / np.abs(h) ** 2


def main():
    from neuro_py.io import loading
    from scipy.signal import welch
    W = pd.DataFrame(pd.read_pickle(PKL)["windows"])
    W = W[W.nonoverlap & W.d2.notna()].reset_index(drop=True)
    bp = next(b for b in FIG1_SESSIONS if "_".join(session_id(b)) == EX)
    st, cm = loading.load_spikes(bp, brainRegion="CA1")
    sp = np.sort(np.concatenate([np.asarray(u, float) for u in np.asarray(st.data, dtype=object)]))   # all CA1 units

    rows, kern, series = [], [], {}
    for i, w in W.iterrows():
        x = getFR(sp, w.center_s - WIN / 2, w.center_s + WIN / 2)
        phi, s2 = yw_kernel(x)
        kern.append(phi); series[i] = x
        rows.append(dict(session=EX, center_s=w.center_s, state=w.state, d2_stored=w.d2, sum_phi=phi.sum(),
                         d2_refit=abs(1 - phi.sum()) / np.sqrt(P), s2=s2))
    D = pd.DataFrame(rows); K = np.array(kern)
    ok = np.corrcoef(D.d2_refit, D.d2_stored)[0, 1]
    dd = np.abs(D.d2_refit - D.d2_stored); exact = int((dd < 1e-6).sum())
    print(f"{EX}: {len(D)} windows; refit vs stored d2 (2024 Euclidean, p = 20): r = {ok:.5f}, "
          f"identical (|diff| < 1e-6) in {exact}/{len(D)}, max |diff| = {dd.max():.2e}")
    print(D.assign(diff=dd)[dd >= 1e-6][['center_s', 'state', 'd2_stored', 'd2_refit']].to_string())
    assert np.median(dd) < 1e-6, "refit does not reproduce the stored d2 -- stop"
    m = D.state.isin(SHOW).to_numpy()
    series = {j: series[i] for j, i in enumerate(np.flatnonzero(m))}
    D, K, dd = D[m].reset_index(drop=True), K[m], dd[m]
    exact = int((dd < 1e-6).sum())
    print(f"shown: {', '.join(SHOW)}, {len(D)} windows, refit identical in {exact}")

    # geometry: signed distance along the unit normal, and the in-plane principal directions
    nrm = np.ones(P) / np.sqrt(P)
    D["u"] = (K.sum(1) - 1) / np.sqrt(P)                     # d2 = |u|, u < 0 on the stationary side
    Kp = K - np.outer(K @ nrm, nrm)
    Kc = Kp - Kp.mean(0)
    _, sv, Vt = np.linalg.svd(Kc, full_matrices=False)
    D["x"], D["y"] = Kc @ Vt[0], Kc @ Vt[1]
    print(f"in-plane variance explained by the 2 directions shown: {100 * (sv[:2] ** 2).sum() / (sv ** 2).sum():.0f} %")
    D.to_csv(assert_local_path(os.path.join(LOG, "r38_hyperplane_windows.csv")), index=False)

    ex = {}
    for s in SHOW:   # example = the exact-reproduction window nearest the centre of the cloud in (x, y, u)
        g = D[(D.state == s) & (np.abs(D.d2_refit - D.d2_stored) < 1e-6)]
        c = D[D.state == s][["x", "y", "u"]]
        zc = (g[["x", "y", "u"]] - c.median()) / (c.quantile(0.75) - c.quantile(0.25))
        ex[s] = int((zc ** 2).sum(1).idxmin())
    for s, i in ex.items():
        print(f"example {s}: centre {D.center_s[i]:.0f} s, sum(phi) {D.sum_phi[i]:.3f}, d2 {D.d2_refit[i]:.4f} "
              f"(state median {D[D.state == s].d2_refit.median():.4f}; percentile of d2 "
              f"{100 * (D[D.state == s].d2_refit < D.d2_refit[i]).mean():.0f})")

    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.45], hspace=0.38, wspace=0.32)
    # a: the data
    ga = gs[0, 0].subgridspec(len(ex), 1, hspace=0.75)
    for j, (s, i) in enumerate(ex.items()):
        ax = fig.add_subplot(ga[j])
        x = series[i]; t = np.arange(x.size) * DT
        ax.plot(t, x, color=COL[s], lw=0.6)
        ax.set_xlim(0, WIN); ax.set_ylabel("spikes / 40 ms")
        ax.set_title(("a  " if j == 0 else "") + f"{NAME[s]}: 30 s of population activity", loc="left", fontsize=9.5)
        if j == len(ex) - 1:
            ax.set_xlabel("time in window (s)")
    # b: kernels
    ax = fig.add_subplot(gs[0, 1])
    lags = 1000 * DT * np.arange(1, P + 1)
    for s, i in ex.items():
        ax.plot(lags, K[i], "o-", ms=3.5, color=COL[s],
                label=f"{NAME[s]}: Σφ = {D.sum_phi[i]:.2f}, d2 = |1 − Σφ|/√20 = {D.d2_refit[i]:.3f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("lag (ms)"); ax.set_ylabel("history kernel φ_k")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("b  what is fitted: x_t = Σ φ_k x_(t−k) + noise, k = 1…20\n    (40 ms bins → 800 ms of history)",
                 loc="left", fontsize=9.5)
    # c: spectra
    ax = fig.add_subplot(gs[0, 2])
    f = np.logspace(np.log10(1 / WIN), np.log10(0.5 / DT), 400)      # window length .. Nyquist (12.5 Hz)
    for s, i in ex.items():
        x = series[i]; z = (x - x.mean()) / x.std(ddof=1)
        fw, pw = welch(z, fs=1 / DT, nperseg=250)
        ax.loglog(fw[1:], pw[1:], ".", ms=3, color=COL[s], alpha=0.6, label="the window's own spectrum (Welch)")
        ax.loglog(f, model_psd(K[i], D.s2[i], f), "-", color=COL[s], lw=1.8, label=f"{NAME[s]}, fitted model")
        on = K[i] + (1 - K[i].sum()) / P                     # nearest model on the hyperplane
        ax.loglog(f, model_psd(on, D.s2[i], f), "--", color=COL[s], lw=1.2, label=f"{NAME[s]}, nearest model on the plane")
    i0 = ex["NREM"]; on0 = K[i0] + (1 - K[i0].sum()) / P
    fa = 0.05
    ax.annotate("on the plane:\nslope −2 (β = 2)", xy=(fa, model_psd(on0, D.s2[i0], np.array([fa]))[0]),
                xytext=(0.15, 6), fontsize=8, arrowprops=dict(arrowstyle="-", color="0.3", lw=0.7))
    ax.text(0.036, model_psd(K[i0], D.s2[i0], np.array([0.036]))[0] * 0.3, "off the plane: flat (β = 0)", fontsize=8)
    ax.set_xlim(1 / WIN, 0.5 / DT)
    ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("power of the population spike count\n(per Hz, z-scored; not LFP)")
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("c  what β = 2 means: on the plane the spectrum\n    keeps rising as 1/f²; off it, it flattens",
                 loc="left", fontsize=9.5)
    # d: geometry
    ax = fig.add_subplot(gs[1, :2], projection="3d", computed_zorder=False)
    lim = np.percentile(np.abs(np.r_[D.x, D.y]), 99) * 1.1
    gx, gy = np.meshgrid(np.linspace(-lim, lim, 2), np.linspace(-lim, lim, 2))
    ax.plot_surface(gx, gy, 0 * gx, color="#bdbdbd", alpha=0.35, lw=0)
    ax.set_box_aspect((2.3, 1.3, 1.1))
    ax.text2D(0.60, 0.80, "β = 2 hyperplane: Σφ = 1", transform=ax.transAxes, fontsize=9.5, color="0.2")
    for s in SHOW:
        g = D[D.state == s]
        if len(g):
            ax.scatter(g.x, g.y, g.u, s=7, color=COL[s], alpha=0.6, depthshade=False, zorder=1,
                       label=f"{NAME[s]} ({len(g)} windows, median d2 {g.d2_refit.median():.3f})")
    for s, i in ex.items():
        ax.plot([D.x[i]] * 2, [D.y[i]] * 2, [D.u[i], 0], color=RED, lw=2.2, zorder=9)
        ax.scatter([D.x[i]], [D.y[i]], [0], s=25, color=RED, marker="x", depthshade=False, zorder=9)
        ax.scatter([D.x[i]], [D.y[i]], [D.u[i]], s=90, color=RED, edgecolor="k", lw=0.8, depthshade=False, zorder=10,
                   label="the window in a–c")
        ax.text(D.x[i], D.y[i], D.u[i] / 2, f"   d2 = {D.d2_refit[i]:.3f}", fontsize=9, fontweight="bold", color=RED,
                zorder=11)
    ax.set_zlim(D.u.min() * 1.05, 0.01)
    ax.set_xlabel("in-plane direction 1", labelpad=2); ax.set_ylabel("in-plane direction 2", labelpad=2)
    ax.set_zlabel("(Σφ − 1) / √20", labelpad=8)
    ax.set_xticklabels([]); ax.set_yticklabels([])
    ax.view_init(elev=20, azim=-55)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(-0.12, 0.98))
    ax.set_title(f"d  every 30 s window of {EX} as one point in the space of 20-lag kernels:\n"
                 "    d2 is its perpendicular distance to the plane (red: the window in a–c)", loc="left", fontsize=9.5)
    # e: ruler
    ax = fig.add_subplot(gs[1, 2])
    bins = np.linspace(D.sum_phi.min() - 0.02, 1.0, 40)
    for s in SHOW:
        g = D[D.state == s]
        if len(g):
            ax.hist(g.sum_phi, bins=bins, color=COL[s], alpha=0.75, label=NAME[s], density=True)
    ax.axvline(1, color="k", lw=1.5)
    for s_, i_ in ex.items():
        ax.axvline(D.sum_phi[i_], color=RED, lw=1.8, label="the window in a–c")
    ax.text(0.985, ax.get_ylim()[1] * 0.97, "β = 2 plane", ha="right", va="top", fontsize=8)
    ax.set_xlabel("Σφ (sum of the 20 kernel weights)"); ax.set_ylabel("windows (density)")
    top = ax.secondary_xaxis("top", functions=(lambda v: np.abs(1 - v) / np.sqrt(P), lambda d: 1 - d * np.sqrt(P)))
    top.set_xlabel("d2 = |1 − Σφ| / √20")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("e  the ruler", loc="left", fontsize=9.5)
    fig.text(0.01, 0.005, f"{EX}, all CA1 units, non-overlapping 30 s {'/'.join(SHOW)} windows; AR(20) by Yule–Walker on "
             f"z-scored 40 ms counts, the published recipe (refit identical to the stored d2 in {exact} of {len(D)} windows)." f" Example: the window nearest "
             "the centre of the cloud in d. c: points = the window's own spectrum (Welch, 10 s segments); dashed = the kernel "
             "moved perpendicularly onto the plane; frequency axis from 1 / 30 s (the window length) to 12.5 Hz (Nyquist for 40 ms bins). d: axes are exact for p = 20; the two horizontal directions span the "
             "largest in-plane variation and do not affect d2.", fontsize=7.5, color="#555555", wrap=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{OUT}.{ext}"), dpi=250, bbox_inches="tight")
    print(f"-> {OUT}.png/.svg")


if __name__ == "__main__":
    main()
