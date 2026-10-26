"""r35_fig2d_replot.py -- d2 against replay probability and the graded replay measures, on
non-overlapping windows.

Windows: the replay-probability tables (data/replay/replay_probality/<session>.pkl, 30 s
windows stepped 2 s), NREM windows with at least one ripple, thinned to the non-overlapping
subset (every 15th window, 30 s apart).

replay_prob says how often replay happens per ripple; the per-event measures say how much
each replay event carries. Like replay_prob they are intensive (per-event means), so the total
SWR rate divides out:

  dur_s        duration of the significant decoded replay event (s)
  traj_span    decoded positions covered, max - min of the argmax posterior (bins)
  traj_net     net displacement, |last - first| (bins)
  speed        |slope| of the fitted trajectory
  rval         sequence rank correlation (graded replay quality)
  gof          decoder goodness of fit

Per window, each measure is the mean over the significant replay events overlapping it, so
windows with no replay contribute to replay_prob (as 0) but not to the per-event measures; n
differs by column and is reported.

d2 = d2 (2024 Euclidean, p = 20). Session is the unit of analysis: per-session Spearman,
also partialling out the ripple count, Fisher-z mean and signed-rank test across the 9
sessions. window_means, partial_spearman, fisher and MEASURES are used by
r35_replay_geometry_excised.py.

    -> data/revision/log/r35_fig2d_replay_measures.csv
    -> data/revision/log/r35_fig2d_replay_measures_summary.csv
    -> reports/revision_figs/fig2d_nonoverlap_replay_measures.png / .svg
"""
from __future__ import annotations

import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r35_swr_property_control import load_replay_events  # noqa: E402

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
PROB_DIR = os.path.join(REPO, "data", "replay", "replay_probality")
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
WIN_S = 30.0
NONOVL = 15
N_BINS = 6
STATE = "NREM"

MEASURES = [
    ("replay_prob", "replay probability", "replay events / ripple"),
    ("dur_s",       "replay duration",    "s"),
    ("traj_span",   "trajectory span",    "position bins"),
    ("traj_net",    "net displacement",   "position bins"),
    ("speed",       "trajectory speed",   "|slope|"),
    ("rval",        "sequence quality r", "rank corr"),
    ("gof",         "goodness of fit",    ""),
]


def assert_local_path(path):
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    if os.path.splitdrive(p)[0].rstrip(":").upper() not in {"C", "D"}:
        raise PermissionError(f"refusing to write outside C:/D:: {p}")
    return p


def zscore(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / sd if np.isfinite(sd) and sd > 0 else np.full_like(x, np.nan)


def window_means(ev, vals, centers, win=WIN_S):
    """Mean of `vals` over events overlapping each window; NaN where none."""
    out = np.full(len(centers), np.nan)
    if not len(ev):
        return out
    a, b = ev[:, 0], ev[:, 1]
    for i, c in enumerate(centers):
        m = (a < c + win / 2) & (b > c - win / 2)
        if m.any():
            v = vals[m]
            v = v[np.isfinite(v)]
            if v.size:
                out[i] = v.mean()
    return out


def build(session):
    with open(os.path.join(PROB_DIR, f"{session}.pkl"), "rb") as fh:
        o = pickle.load(fh)
    d = o["per_center"]
    d = d[d["state"].astype(str).str.upper().eq(STATE)]
    d = d[d["ripple_count"].astype(float) > 0]
    d = d.dropna(subset=["d2", "replay_prob"]).sort_values("center_s")
    d = d.iloc[::NONOVL].reset_index(drop=True)          # non-overlapping
    c = d["center_s"].to_numpy(float)

    E = load_replay_events(session)
    S = E[E.is_sig].copy()
    S["dur_s"] = S["stop"] - S["start"]
    S["speed"] = S["slope"].abs()
    ev = S[["start", "stop"]].to_numpy()

    out = pd.DataFrame({"session": session, "center_s": c,
                        "d2": d["d2"].to_numpy(float),
                        "ripple_count": d["ripple_count"].to_numpy(float),
                        "replay_prob": d["replay_prob"].to_numpy(float)})
    for col in ["dur_s", "traj_span", "traj_net", "speed", "rval", "gof"]:
        out[col] = window_means(ev, S[col].to_numpy(float), c)
    return out


def partial_spearman(y, x, z):
    """Spearman partial correlation of y and x given covariate z."""
    ok = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if ok.sum() < 10:
        return np.nan
    ry, rx, rz = (stats.rankdata(v[ok]) for v in (y, x, z))
    A = np.column_stack([np.ones(rz.size), rz])
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    if ey.std() == 0 or ex.std() == 0:
        return np.nan
    return float(np.corrcoef(ey, ex)[0, 1])


def fisher(v):
    v = np.asarray([x for x in np.asarray(v, float) if np.isfinite(x)], float)
    if v.size == 0:
        return np.nan, np.nan, 0
    z = np.arctanh(np.clip(v, -0.999999, 0.999999))
    return (float(np.tanh(z.mean())),
            float(z.std(ddof=1) / np.sqrt(z.size)) if z.size > 1 else np.nan, z.size)


def main():
    sessions = sorted(f[:-4] for f in os.listdir(PROB_DIR) if f.endswith(".pkl"))
    print(f"Fig. 2d session set ({len(sessions)}): {', '.join(sessions)}\n")
    parts = []
    for s in sessions:
        try:
            parts.append(build(s))
            w = parts[-1]
            print(f"  {s:12} {len(w):4d} non-overlapping windows, "
                  f"{int(w.dur_s.notna().sum()):4d} with >=1 replay event", flush=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"  [SKIP] {s}: {type(e).__name__}: {e}", flush=True)
    W = pd.concat(parts, ignore_index=True)
    assert_local_path(LOGDIR); os.makedirs(LOGDIR, exist_ok=True)
    W.to_csv(assert_local_path(os.path.join(
        LOGDIR, "r35_fig2d_replay_measures.csv")), index=False)

    print(f"\n{'='*78}\nd2 (2024 Euclidean, p=20) ~ replay measures, "
          f"NON-OVERLAPPING windows, n={W.session.nunique()} sessions\n{'='*78}")
    # d2 also tracks ripple_count (+0.26), and events packed closer together could be
    # mechanically shorter -- so every measure is also reported partialling it out.
    print(f"{'measure':22}{'r':>9}{'se':>8}{'p':>9}{'pos':>7}"
          f"{'r|ripples':>11}{'p':>8}{'med n':>7}")
    rows = []
    for col, lbl, unit in MEASURES:
        rs, ps_, ns = [], [], []
        for s, g in W.groupby("session"):
            v = g[["d2", col, "ripple_count"]].dropna()
            ns.append(len(v))
            ok = len(v) >= 10 and v[col].nunique() > 2
            rs.append(stats.spearmanr(v["d2"], v[col]).statistic if ok else np.nan)
            ps_.append(partial_spearman(v["d2"].to_numpy(), v[col].to_numpy(),
                                        v["ripple_count"].to_numpy())
                       if ok else np.nan)
        r, se, n = fisher(rs)
        rp, sep, _ = fisher(ps_)
        vv = np.asarray([x for x in rs if np.isfinite(x)])
        pv = np.asarray([x for x in ps_ if np.isfinite(x)])
        p = stats.wilcoxon(vv).pvalue if vv.size >= 3 else np.nan
        pp = stats.wilcoxon(pv).pvalue if pv.size >= 3 else np.nan
        print(f"{lbl:22}{r:+9.3f}{se:8.3f}{p:9.4f}"
              f"{int((vv>0).sum())}/{vv.size:<4}{rp:+11.3f}{pp:8.4f}"
              f"{int(np.median(ns)):>7}")
        rows.append(dict(measure=col, label=lbl, unit=unit, r=r, se=se, p=p,
                         r_partial_ripplecount=rp, se_partial=sep, p_partial=pp,
                         n_pos=int((vv > 0).sum()), n_sessions=vv.size,
                         median_n_windows=int(np.median(ns))))
    pd.DataFrame(rows).to_csv(assert_local_path(os.path.join(
        LOGDIR, "r35_fig2d_replay_measures_summary.csv")), index=False)
    plot(W)


def sextile_curves(W, col):
    """Per session: equal-count d2 sextiles, mean z(col) in each."""
    cur = []
    for s, g in W.groupby("session"):
        g = g[["d2", col]].dropna()
        if len(g) < 3 * N_BINS or g[col].nunique() <= 2:
            continue
        t = pd.qcut(g["d2"], q=N_BINS, labels=False, duplicates="drop") + 1
        z = pd.Series(zscore(g[col].to_numpy()), index=g.index)
        cur.append(z.groupby(t).mean().reindex(range(1, N_BINS + 1)).to_numpy())
    return np.asarray(cur, float)


def plot(W):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    show = ["replay_prob", "dur_s", "traj_span", "rval"]
    lbl = {c: l for c, l, _ in MEASURES}
    fig, axes = plt.subplots(1, len(show) + 1, figsize=(4.0 * (len(show) + 1) / 1.35, 3.7))
    x = np.arange(1, N_BINS + 1)
    for ax, col in zip(axes[:len(show)], show):
        C = sextile_curves(W, col)
        if not C.size:
            continue
        for row in C:
            ax.plot(x, row, color="0.84", lw=0.9, zorder=1)
        m = np.nanmean(C, 0)
        se = np.nanstd(C, 0, ddof=1) / np.sqrt(np.sum(np.isfinite(C), 0))
        ax.errorbar(x, m, yerr=se, fmt="o-", color="#691c1b", lw=2.2, ms=7,
                    capsize=4, zorder=3)
        ax.axhline(0, color="0.85", lw=0.8, zorder=0)
        ax.set_xlabel("d2 sextile (low $\\rightarrow$ high)")
        ax.set_title(f"{lbl[col]}\nn={C.shape[0]} sessions", fontsize=9)
    axes[0].set_ylabel("z within session")

    ax = axes[-1]
    rs = []
    for col, l, _ in MEASURES:
        v = [stats.spearmanr(g[["d2", col]].dropna()["d2"],
                             g[["d2", col]].dropna()[col]).statistic
             for _, g in W.groupby("session")
             if len(g[["d2", col]].dropna()) >= 10]
        rs.append((l, *fisher(v)[:2]))
    lbls = [r[0] for r in rs]
    vals = np.array([r[1] for r in rs])
    errs = np.array([r[2] for r in rs])
    y = np.arange(len(rs))
    ax.barh(y, vals, xerr=errs, color=np.where(vals > 0, "#691c1b", "#2c3e50"),
            error_kw=dict(lw=1.1))
    ax.set_yticks(y)
    ax.set_yticklabels(lbls, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="0.3", lw=0.9)
    ax.set_xlabel("Spearman r with d2\n(Fisher-z mean $\\pm$ SEM)")
    ax.set_title("all replay measures\nnon-overlapping windows", fontsize=9)

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    assert_local_path(FIGDIR); os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(
            FIGDIR, f"fig2d_nonoverlap_replay_measures.{ext}")), dpi=200,
            bbox_inches="tight")
    print(f"\nfigure -> {FIGDIR}\\fig2d_nonoverlap_replay_measures.png/.svg")
    print(f"tables -> {LOGDIR}\\r35_fig2d_replay_measures*.csv")


if __name__ == "__main__":
    main()
