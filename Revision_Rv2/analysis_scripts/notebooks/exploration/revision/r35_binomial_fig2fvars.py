"""
r35_binomial_fig2fvars.py -- the binomial replay model with Figure 2F's variable set,
under the published AND the ripple-excised d2.

Figure 2F's target (replay present in a 30 s window) is close to saturated across most
windows, leaving its interaction ablations little room to move under either d2. The
binomial model (r35_binomial_replay.py) is the well-posed alternative -- per-ripple replay
odds with SWR abundance as the number of trials -- but its covariate ladder used d2 / +MUA
/ +SWR rate rather than Figure 2F's own variable set, and had not been run with the
excised d2.

This script keeps everything from `r35_binomial_replay.py` that matters -- the same
window selection (NREM, ripple_count > 0, every 15th window of the Fig. 2d grid so
windows do not overlap), the same fitting functions (imported, not copied), the same
within-session z-scoring, the same session-as-unit signed-rank -- and changes three
things:

  1. COVARIATES are Fig 2F's: d2, MUA, MUA_std, EI_ratio, read from Fig 2F's own
     per-window table (`replay_probality/<session>.pkl["per_center"]`), so this is the
     same variable set on the same windows the panel used.
  2. LADDER:  M1 d2 | M2 d2 + MUA | M3 d2 + MUA + MUA_std | M4 d2 + MUA + MUA_std + EI_ratio.
     SWR rate is not in the main ladder -- it is already the binomial denominator, and
     entering it twice over-controls. A sub-ladder (M2r, M3r, M5, R0) adds it anyway,
     reported as a follow-up check.
  3. d2 ARMS: `published` (2024 Euclidean, unexcised) and `excised` (ripple-excised),
     both read from the .mat files via r35_background_vs_ripple.load_d2_mat and joined
     on center_s. The excised d2 is measured largely independently of MUA, unlike the
     published one.

Each row gives the per-ripple odds that a SWR carries replay as a function of d2,
holding firing rate (and its variability, and E/I) fixed.

Sessions: the replay sessions (SESSIONS in r35_swr_property_control.py). PPP23_day8's
pickle has no MUA/MUA_std/EI_ratio and is dropped by name.

    -> data/revision/log/r35_binomial_fig2fvars.csv          (b1 per arm x model x session)
    -> data/revision/log/r35_binomial_fig2fvars_windows.csv  (the window table)
    -> reports/revision_figs/r35_binomial_fig2fvars.png / .svg
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
from run_fig1_metrics import assert_local_path                       # noqa: E402
from r35_swr_property_control import SESSIONS                        # noqa: E402
from r35_background_vs_ripple import load_d2_mat                     # noqa: E402
from r35_binomial_replay import fit_binomial, pooled_fit, zs, NONOVL, WIN_S  # noqa: E402

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
PROB_DIR = os.path.join(REPO, "data", "replay", "replay_probality")
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
JOIN_TOL_S = 1.0
COVS = ("MUA", "MUA_std", "EI_ratio")
ARMS = [("published", ""), ("excised", "_rippleablate")]
MODELS = [
    ("M1", ["d2_z"],                                     "d2"),
    ("M2", ["d2_z", "MUA_z"],                            "d2 + MUA"),
    ("M3", ["d2_z", "MUA_z", "MUA_std_z"],               "d2 + MUA + MUA_std"),
    ("M4", ["d2_z", "MUA_z", "MUA_std_z", "EI_ratio_z"], "d2 + MUA + MUA_std + EI_ratio"),
    # SWR-rate sub-ladder, conditioning on SWR rate in addition to the binomial
    # denominator. Rate is already the denominator, so these over-condition; reported
    # as a follow-up check.
    ("M2r", ["d2_z", "rate_z"],                          "d2 + SWR rate"),
    ("M3r", ["d2_z", "MUA_z", "rate_z"],                 "d2 + MUA + SWR rate"),
    ("M5",  ["d2_z", "MUA_z", "MUA_std_z", "rate_z"],    "d2 + MUA + MUA_std + SWR rate"),
    ("R0",  ["rate_z"],                                  "SWR rate alone (b1 = rate)"),
    # fully cumulative rung, so the boxplot ladder (fig2f_binomial_boxplot.py) reads
    # left-to-right without a gap
    ("M6",  ["d2_z", "MUA_z", "MUA_std_z", "rate_z", "EI_ratio_z"],
            "d2 + MUA + MUA_std + SWR rate + EI_ratio"),
]
PLOT_MODELS = MODELS[:4]   # the figure keeps the main ladder; rate models are text/CSV only


def _nearest(centers, m):
    mc = m["center"].to_numpy(float)
    mv = m["d2_x"].to_numpy(float)
    idx = np.clip(np.searchsorted(mc, centers), 0, len(mc) - 1)
    lo = np.clip(idx - 1, 0, len(mc) - 1)
    pick = np.where(np.abs(mc[lo] - centers) < np.abs(mc[idx] - centers), lo, idx)
    out = mv[pick].copy()
    out[np.abs(mc[pick] - centers) > JOIN_TOL_S] = np.nan
    return out


def load():
    out = []
    for sess, bp in SESSIONS.items():
        o = pd.read_pickle(os.path.join(PROB_DIR, f"{sess}.pkl"))
        d = o["per_center"]
        d = d[d["state"].astype(str).str.upper().eq("NREM")]
        missing = [c for c in COVS if c not in d.columns]
        if missing:
            print(f"  [drop] {sess}: no {missing} in its table -- not reconstructed")
            continue
        d = d[d["ripple_count"].astype(float) > 0]
        d = d.dropna(subset=["replay_count", "ripple_count", *COVS]).sort_values("center_s")
        d = d.iloc[::NONOVL].reset_index(drop=True)        # non-overlapping, as the parent
        c = d["center_s"].to_numpy(float)
        W = pd.DataFrame({"session": sess, "center_s": c,
                          "n_swr": d["ripple_count"].to_numpy(float).round().astype(int),
                          "n_replay": d["replay_count"].to_numpy(float).round().astype(int)})
        for k in COVS:
            W[k] = d[k].to_numpy(float)
        for arm, tag in ARMS:
            m = load_d2_mat(bp, tag)
            W[f"d2_{arm}"] = np.nan if m is None or not len(m) else _nearest(c, m.sort_values("center"))
        W = W[W["n_swr"] > 0].reset_index(drop=True)
        W["n_replay"] = np.minimum(W["n_replay"], W["n_swr"])
        print(f"  {sess:12s} {len(W):4d} windows | d2 joined: "
              + ", ".join(f"{a} {W[f'd2_{a}'].notna().mean():.0%}" for a, _ in ARMS), flush=True)
        out.append(W)
    D = pd.concat(out, ignore_index=True)
    # SWR rate as a covariate too. It is already the binomial denominator, so this
    # over-conditions; reported as a follow-up check.
    D["rate"] = D["n_swr"] / WIN_S
    D["rate_z"] = D.groupby("session")["rate"].transform(zs)
    for k in COVS:
        D[f"{k}_z"] = D.groupby("session")[k].transform(zs)
    for arm, _ in ARMS:
        D[f"d2_{arm}_z"] = D.groupby("session")[f"d2_{arm}"].transform(zs)
    return D


def main():
    D = load()
    assert_local_path(LOGDIR)
    os.makedirs(LOGDIR, exist_ok=True)
    print(f"\n{D.session.nunique()} sessions, {len(D)} non-overlapping NREM windows, "
          f"{int(D.n_swr.sum())} SWRs, {int(D.n_replay.sum())} with replay "
          f"({D.n_replay.sum() / D.n_swr.sum():.1%})")
    for arm, _ in ARMS:
        r = D[[f"d2_{arm}", "MUA"]].dropna().corr().iloc[0, 1]
        print(f"  r(d2_{arm}, MUA) across windows = {r:+.3f}")

    rows = []
    for arm, _ in ARMS:
        Da = D.copy()
        Da["d2_z"] = Da[f"d2_{arm}_z"]
        for tag, cols, lbl in MODELS:
            for s, g in Da.groupby("session"):
                r = fit_binomial(g, cols)
                if r is not None:
                    rows.append(dict(arm=arm, model=tag, label=lbl, session=s, **r))
            pf = pooled_fit(Da, cols)
            if pf:
                rows.append(dict(arm=arm, model=tag, label=lbl, session="__pooled__", **pf))
    # Head-to-head with EI_ratio, to separate collinearity from E/I being the stronger
    # predictor at the point the ladder adds it: fit each alone and each conditioned on
    # the other (b1 is always the FIRST column in fit_binomial).
    for arm, _ in ARMS:
        Da = D.copy()
        Da["d2_z"] = Da[f"d2_{arm}_z"]
        for cols, lbl in ((["EI_ratio_z"], "EI_ratio alone"),
                          (["EI_ratio_z", "d2_z"], "EI_ratio | d2"),
                          (["d2_z", "EI_ratio_z"], "d2 | EI_ratio")):
            for s, g in Da.groupby("session"):
                r = fit_binomial(g, cols)
                if r is not None:
                    rows.append(dict(arm=arm, model="H2H", label=lbl, session=s, **r))
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(os.path.join(LOGDIR, "r35_binomial_fig2fvars.csv")), index=False)
    print("\n--- within-session r(d2, covariate), median over sessions ---")
    for arm, _ in ARMS:
        rs = {k: np.median([g[[f"d2_{arm}", k]].corr().iloc[0, 1]
                            for _, g in D.groupby("session")]) for k in COVS}
        print(f"  d2_{arm:9s} " + "  ".join(f"{k} {v:+.3f}" for k, v in rs.items()))
    print("  r(EI_ratio, MUA) = "
          f"{np.median([g[['EI_ratio', 'MUA']].corr().iloc[0, 1] for _, g in D.groupby('session')]):+.3f}")
    D.to_csv(assert_local_path(os.path.join(LOGDIR, "r35_binomial_fig2fvars_windows.csv")), index=False)
    report(R)
    report_h2h(R)
    plot(R)


def report(R):
    print(f"\n{'=' * 90}\nEFFECT OF d2 ON PER-RIPPLE ODDS OF REPLAY -- b1 per +1 within-session SD of d2")
    print("session = unit (signed-rank across sessions); pooled = session FE, cluster-robust SE")
    print("=" * 90)
    for arm, _ in ARMS:
        print(f"\n--- d2 = {arm.upper()} ---")
        print(f"  {'model':34}{'median b1':>11}{'p':>9}{'pos':>7}{'OR/SD':>8}{'pooled b1':>12}{'p':>9}")
        for tag, cols, lbl in MODELS:
            P = R[(R.arm == arm) & (R.model == tag) & (R.session != "__pooled__")]
            Q = R[(R.arm == arm) & (R.model == tag) & (R.session == "__pooled__")]
            if len(P) < 3:
                continue
            b = P.b1.to_numpy()
            pw = stats.wilcoxon(b).pvalue
            pb = f"{Q.b1.iloc[0]:+12.4f}{Q.p.iloc[0]:9.2g}" if len(Q) else ""
            print(f"  {tag + ': ' + lbl:34}{np.median(b):+11.4f}{pw:9.4f}"
                  f"{int((b > 0).sum()):>4}/{len(b):<3}{np.exp(np.median(b)):8.3f}{pb}")


def report_h2h(R):
    print(f"\n{'=' * 90}\nHEAD-TO-HEAD WITH EI_ratio (why M4 collapses)\n{'=' * 90}")
    for arm, _ in ARMS:
        print(f"--- d2 = {arm} ---")
        for lbl in ("EI_ratio alone", "EI_ratio | d2", "d2 | EI_ratio"):
            P = R[(R.arm == arm) & (R.model == "H2H") & (R.label == lbl)]
            if len(P) < 3:
                continue
            b = P.b1.to_numpy()
            print(f"  {lbl:16} b(first) median {np.median(b):+.4f}  "
                  f"pos {int((b > 0).sum())}/{len(b)}  p = {stats.wilcoxon(b).pvalue:.4f}  "
                  f"OR/SD {np.exp(np.median(b)):.3f}")


def plot(R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 10, "axes.linewidth": 1.2})
    col = {"published": "0.45", "excised": "#7b3294"}
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    x0 = np.arange(len(PLOT_MODELS))
    rng = np.random.default_rng(0)
    for j, (arm, _) in enumerate(ARMS):
        off = (j - 0.5) * 0.3
        for i, (tag, _, _) in enumerate(PLOT_MODELS):
            P = R[(R.arm == arm) & (R.model == tag) & (R.session != "__pooled__")]
            b = P.b1.to_numpy()
            if not len(b):
                continue
            ax.plot(x0[i] + off + rng.normal(0, 0.03, len(b)), b, "o", ms=4,
                    color=col[arm], alpha=0.55, zorder=2)
            ax.plot([x0[i] + off - 0.09, x0[i] + off + 0.09], [np.median(b)] * 2,
                    color=col[arm], lw=3, zorder=3,
                    label=f"d2 = {arm}" if i == 0 else None)
            pw = stats.wilcoxon(b).pvalue
            star = "***" if pw < .001 else "**" if pw < .01 else "*" if pw < .05 else "ns"
            ax.text(x0[i] + off, ax.get_ylim()[1] if False else np.max(b) + 0.006, star,
                    ha="center", va="bottom", fontsize=8, color=col[arm])
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.set_xticks(x0)
    ax.set_xticklabels([f"{t}\n{l}" for t, _, l in PLOT_MODELS], fontsize=8)
    ax.set_ylabel("b1: log-odds of replay per +1 SD d2\n(per session; bar = median)")
    ax.set_title("Binomial replay model with Fig 2F's covariates — does d2 survive "
                 "conditioning on MUA, MUA_std, E/I?", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    assert_local_path(FIGDIR)
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"r35_binomial_fig2fvars.{ext}")),
                    dpi=300, bbox_inches="tight")
    print(f"\nfigure -> {FIGDIR}\\r35_binomial_fig2fvars.png/.svg")


if __name__ == "__main__":
    main()
