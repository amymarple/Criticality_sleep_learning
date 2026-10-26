"""
r35_binomial_replay.py -- binomial model of whether a SWR carries replay, as a function
of d2, replacing the ratio correlation.

Fig. 2d correlates `replay_prob = replay_count / ripple_count`
against d2. That treats a window with 1 replay out of 2 SWRs as equal evidence to 10 out
of 20 -- same ratio, different precision. The binomial likelihood carries the
difference:

    replay_count_w ~ Binomial( N_SWR_w , p_w )
    logit p_w = b0 + b1 * d2_w   ( + covariates )   , session as a grouping factor

It also avoids the Fig. 2e rate contrast, where `rate_in + rate_out == rate_all` leaves
a two-arm rate contrast not identified, and where a matched-absolute-rate panel would
need to match ranges across the substate structure Chang et al. 2025 (Nature 637:1161)
established: replay-carrying SWRs concentrate in contracted-pupil substates. The
binomial model needs no rate matching: N_SWR enters as the number of trials, so SWR
abundance is conditioned on by construction and b1 is the effect of d2 on the
*per-ripple* odds of replay.

MUA is a covariate of d2 (`r35_glm_ablation.py`); this script fits a
ladder:

    M1 : logit p ~ d2
    M2 : logit p ~ d2 + MUA
    M3 : logit p ~ d2 + MUA + SWR rate

b1 is read off each model in turn, from d2 alone to d2 with firing rate and SWR
abundance included.

STATISTICS. Session is the unit of analysis, so the primary estimate is a per-session
binomial GLM and a signed-rank across sessions on b1. A pooled fit with session fixed
effects and session-clustered robust SE is reported alongside. d2 is z-scored within
session, so b1 is the log-odds change per within-session SD of d2 and is comparable
across sessions. Overdispersion (Pearson chi2/df) is reported because window counts are
not iid Bernoulli.

NREM only, non-overlapping windows (every 15th of the 30 s / 2 s grid), windows with at
least one SWR. Fig. 2d session set (with PPP4_day11).

    -> data/revision/log/r35_binomial_replay.csv
    -> reports/revision_figs/fig2d_binomial_replay.png / .svg
"""
from __future__ import annotations

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
PROB_DIR = os.path.join(REPO, "data", "replay", "replay_probality")
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
MUA_SRC = os.path.join(LOGDIR, "r35_glm_ablation_windows.csv")
NONOVL = 15
WIN_S = 30.0

MODELS = [
    ("M1", ["d2_z"],                       "d2"),
    ("M2", ["d2_z", "MUA_z"],              "d2 + MUA"),
    ("M3", ["d2_z", "MUA_z", "rate_z"],    "d2 + MUA + SWR rate"),
]


def assert_local_path(path):
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    if os.path.splitdrive(p)[0].rstrip(":").upper() not in {"C", "D"}:
        raise PermissionError(f"refusing to write outside C:/D:: {p}")
    return p


def zs(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / (sd if np.isfinite(sd) and sd > 0 else 1.0)


def mua_from_spikes(basepath, centers):
    """Population CA1 rate in each 30 s window, straight from the spikes."""
    from neuro_py.io import loading
    st, _ = loading.load_spikes(basepath, brainRegion="CA1")
    units = [np.asarray(u, float) for u in np.asarray(st.data, dtype=object)]
    allsp = np.sort(np.concatenate(units))
    return ((np.searchsorted(allsp, centers + WIN_S / 2)
             - np.searchsorted(allsp, centers - WIN_S / 2)) / WIN_S)


def load(d2_source="euclid", kl_order=20):
    """Windows with integer replay_count / ripple_count, plus MUA.

    d2_source='euclid' : the `d2` column of the replay_probality pickle -- **d2 (2024
                         Euclidean, p=20)**, the published metric. Windows are the Fig. 2d
                         grid subsampled to every 15th (non-overlapping).
    d2_source='kl'     : **d2 (2025 KL)** in bits/s, from
                         `data/revision/log/kl_error_bars.csv` at AR order `kl_order`.
                         Those windows come from the revision pipeline and are ALREADY
                         non-overlapping 30 s windows, so instead of subsampling they are
                         joined onto the full Fig. 2d grid by nearest centre within 1 s
                         (KL window centre = w + 15). Covers the 8 sessions that are
                         also in the Fig-1 18-session set.

    Everything else -- NREM, ripple_count > 0, the replay counts -- is identical, so the
    two arms are directly comparable.
    """
    KL = None
    if d2_source == "kl":
        KL = pd.read_csv(os.path.join(LOGDIR, "kl_error_bars.csv"))
        KL["sess"] = KL["animal"].astype(str) + "_" + KL["day"].astype(str)

    mua = None
    if os.path.exists(MUA_SRC):
        mua = pd.read_csv(MUA_SRC)[["session", "center_s", "MUA"]]
        mua["_k"] = mua["center_s"].round(3)

    out = []
    for f in sorted(os.listdir(PROB_DIR)):
        if not f.endswith(".pkl"):
            continue
        with open(os.path.join(PROB_DIR, f), "rb") as fh:
            o = pickle.load(fh)
        d = o["per_center"]
        d = d[d["state"].astype(str).str.upper().eq("NREM")]
        d = d[d["ripple_count"].astype(float) > 0]
        d = d.dropna(subset=["d2", "replay_count", "ripple_count"])
        d = d.sort_values("center_s")
        sess = f[:-4]

        if d2_source == "kl":
            k = KL[KL["sess"] == sess]
            if not len(k):
                print(f"  [skip] {sess}: no KL windows (not in the Fig-1 18-session set)")
                continue
            col = f"d2mle_p{kl_order}"
            k = (k[["w", col]].assign(center_s=lambda t: t["w"].astype(float) + 15.0)
                 [["center_s", col]].rename(columns={col: "d2_kl"})
                 .dropna().sort_values("center_s"))
            n_kl = len(k)
            d = pd.merge_asof(k, d.drop(columns=["d2"]), on="center_s",
                              tolerance=1.0, direction="nearest")
            d = d.dropna(subset=["d2_kl", "replay_count", "ripple_count"])
            d = d.rename(columns={"d2_kl": "d2"}).reset_index(drop=True)
            print(f"  {sess:12} KL p={kl_order}: {len(d)}/{n_kl} windows joined "
                  f"({len(d)/max(n_kl,1):.0%})")
        else:
            d = d.iloc[::NONOVL].reset_index(drop=True)

        W = pd.DataFrame({
            "session": sess,
            "center_s": d["center_s"].to_numpy(float),
            "d2": d["d2"].to_numpy(float),
            "n_swr": d["ripple_count"].to_numpy(float).round().astype(int),
            "n_replay": d["replay_count"].to_numpy(float).round().astype(int),
        })
        W = W[W["n_swr"] > 0].reset_index(drop=True)
        W["rate"] = W["n_swr"] / WIN_S
        if d2_source == "kl":
            from r35_glm_ablation import BASEPATH
            try:
                W["MUA"] = mua_from_spikes(BASEPATH[sess],
                                           W["center_s"].to_numpy(float))
            except Exception as e:                               # noqa: BLE001
                print(f"    [!] {sess} MUA failed ({type(e).__name__})")
                W["MUA"] = np.nan
        elif mua is not None:
            W["_k"] = W["center_s"].round(3)
            W = W.merge(mua[mua.session == sess][["_k", "MUA"]],
                        on="_k", how="left").drop(columns="_k")
        else:
            W["MUA"] = np.nan
        out.append(W)
    D = pd.concat(out, ignore_index=True)
    D["n_replay"] = np.minimum(D["n_replay"], D["n_swr"])       # guard rounding
    for c, z in (("d2", "d2_z"), ("MUA", "MUA_z"), ("rate", "rate_z")):
        D[z] = D.groupby("session")[c].transform(zs)
    return D


def fit_binomial(sub, cols):
    """Per-session binomial GLM on (successes, failures). Returns b1 for cols[0]."""
    import statsmodels.api as sm
    s = sub[["n_replay", "n_swr"] + cols].dropna()
    if len(s) < 25 or s["n_replay"].sum() == 0:
        return None
    endog = np.column_stack([s["n_replay"], s["n_swr"] - s["n_replay"]])
    X = sm.add_constant(s[cols].to_numpy(float), has_constant="add")
    try:
        r = sm.GLM(endog, X, family=sm.families.Binomial()).fit()
    except Exception:                                            # noqa: BLE001
        return None
    disp = float(r.pearson_chi2 / r.df_resid) if r.df_resid > 0 else np.nan
    return dict(b1=float(r.params[1]), se=float(r.bse[1]), p=float(r.pvalues[1]),
                n=len(s), overdispersion=disp,
                # scale SE by sqrt(overdispersion) -- quasi-binomial correction
                se_qb=float(r.bse[1] * np.sqrt(max(disp, 1.0))))


def pooled_fit(D, cols):
    """Pooled GLM with session fixed effects and session-clustered robust SE."""
    import statsmodels.api as sm
    s = D[["session", "n_replay", "n_swr"] + cols].dropna()
    if not len(s):
        return None
    dummies = pd.get_dummies(s["session"], drop_first=True).astype(float)
    X = pd.concat([s[cols].reset_index(drop=True),
                   dummies.reset_index(drop=True)], axis=1)
    X = sm.add_constant(X, has_constant="add")
    endog = np.column_stack([s["n_replay"], s["n_swr"] - s["n_replay"]])
    try:
        r = sm.GLM(endog, X.to_numpy(float), family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": s["session"].to_numpy()})
    except Exception as e:                                       # noqa: BLE001
        print(f"    [pooled fit failed: {type(e).__name__}]")
        return None
    j = list(X.columns).index(cols[0])
    return dict(b1=float(r.params[j]), se=float(r.bse[j]), p=float(r.pvalues[j]),
                n=len(s))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--d2", choices=("euclid", "kl"), default="euclid",
                    help="euclid = d2 (2024 Euclidean, p=20); kl = d2 (2025 KL) bits/s")
    ap.add_argument("--kl-order", type=int, default=20)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    a = ap.parse_args()

    D = load(a.d2, a.kl_order)
    print(f"d2 source: {a.d2}"
          + (f" (order p={a.kl_order})" if a.d2 == "kl" else ""))
    assert_local_path(LOGDIR)
    print(f"{D.session.nunique()} sessions, {len(D)} non-overlapping NREM windows")
    print(f"  total SWRs {int(D.n_swr.sum())}, replay SWRs {int(D.n_replay.sum())} "
          f"({D.n_replay.sum()/D.n_swr.sum():.1%})")
    print(f"  MUA joined for {int(D.MUA.notna().sum())}/{len(D)} windows")

    rows = []
    for tag, cols, lbl in MODELS:
        print(f"\n{'='*74}\n{tag}: logit p ~ {lbl}\n{'='*74}")
        print(f"  {'session':13}{'b1':>9}{'se':>8}{'se_qb':>8}{'overdisp':>10}{'n':>6}")
        per = []
        for s, g in D.groupby("session"):
            r = fit_binomial(g, cols)
            if r is None:
                print(f"  {s:13}  (skipped)")
                continue
            per.append(dict(model=tag, session=s, **r))
            print(f"  {s:13}{r['b1']:+9.4f}{r['se']:8.4f}{r['se_qb']:8.4f}"
                  f"{r['overdispersion']:10.2f}{r['n']:6d}")
        P = pd.DataFrame(per)
        rows += per
        if len(P) >= 3:
            b = P["b1"].to_numpy()
            w = stats.wilcoxon(b).pvalue
            print(f"\n  per-session b1 (session = unit): median {np.median(b):+.4f}"
                  f"   signed-rank p={w:.4f}   positive in {int((b>0).sum())}/{len(b)}")
            print(f"  odds ratio per +1 SD d2: {np.exp(np.median(b)):.3f}")
        pf = pooled_fit(D, cols)
        if pf:
            print(f"  pooled (session FE, cluster-robust SE): b1={pf['b1']:+.4f} "
                  f"+/- {pf['se']:.4f}, p={pf['p']:.3g}, n={pf['n']} windows")
            rows.append(dict(model=tag, session="__pooled__", **pf))

    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(os.path.join(LOGDIR, f"r35_binomial_replay{a.tag}.csv")),
             index=False)
    D.to_csv(assert_local_path(os.path.join(
        LOGDIR, f"r35_binomial_replay_windows{a.tag}.csv")), index=False)

    print(f"\n{'='*74}\nSUMMARY — effect of d2 on per-ripple odds of replay\n{'='*74}")
    print(f"{'model':26}{'median b1':>11}{'p':>9}{'pos':>7}{'OR/SD':>9}")
    for tag, cols, lbl in MODELS:
        P = R[(R.model == tag) & (R.session != "__pooled__")]
        if len(P) < 3:
            continue
        b = P["b1"].to_numpy()
        print(f"{tag + ': ' + lbl:26}{np.median(b):+11.4f}"
              f"{stats.wilcoxon(b).pvalue:9.4f}"
              f"{int((b>0).sum())}/{len(b):<4}{np.exp(np.median(b)):9.3f}")
    plot(R, D, a.tag, a.d2)


def plot(R, D, tag='', d2src='euclid'):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.8))
    ax = axes[0]
    P = R[(R.model == "M1") & (R.session != "__pooled__")].reset_index(drop=True)
    x = np.arange(len(P))
    ax.bar(x, P.b1, yerr=P.se_qb, capsize=3,
           color=["#8e2420" if v > 0 else "#2c3e50" for v in P.b1],
           error_kw=dict(lw=1.1, ecolor="0.3"))
    med = float(np.median(P.b1))
    ax.axhline(med, color="#111", lw=1.5, ls="--", zorder=4,
               label=f"median {med:+.4f}")
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(P.session, rotation=90, fontsize=7)
    ax.set_ylabel("$\\beta_1$  (log-odds per SD of d2)")
    ax.set_title("M1: logit p ~ %s\nper session, quasi-binomial SE"
                 % ("d2 (2025 KL)" if d2src == "kl" else "d2 (2024 Euclidean)"),
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    # `mtag`, so the caller's `tag` argument (the figure-name suffix) is not shadowed
    for i, (mtag, cols, lbl) in enumerate(MODELS):
        P = R[(R.model == mtag) & (R.session != "__pooled__")]
        if not len(P):
            continue
        ax.scatter(np.full(len(P), i) + np.linspace(-.12, .12, len(P)), P.b1,
                   s=26, color="#691c1b", alpha=0.75)
        ax.plot([i - .22, i + .22], [np.median(P.b1)] * 2, color="#111", lw=2.2)
    ax.axhline(0, color="0.3", lw=0.9)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([m[2].replace(" + ", "\n+ ") for m in MODELS], fontsize=8)
    ax.set_ylabel("$\\beta_1$ for d2")
    ax.set_title("does the d2 effect survive covariates?", fontsize=9)

    ax = axes[2]
    D2 = D.dropna(subset=["d2_z"]).copy()
    D2["bin"] = pd.qcut(D2["d2_z"], 6, labels=False, duplicates="drop") + 1
    g = D2.groupby(["session", "bin"]).agg(r=("n_replay", "sum"),
                                           n=("n_swr", "sum")).reset_index()
    g["p"] = g.r / g.n
    piv = g.pivot(index="bin", columns="session", values="p")
    for c in piv.columns:
        ax.plot(piv.index, piv[c] / piv[c].mean(), color="0.82", lw=0.9, zorder=1)
    m = (piv / piv.mean()).mean(axis=1)
    se = (piv / piv.mean()).std(axis=1, ddof=1) / np.sqrt(piv.shape[1])
    ax.errorbar(m.index, m.values, yerr=se.values, fmt="o-", color="#691c1b",
                lw=2.2, ms=7, capsize=4, zorder=3)
    ax.axhline(1, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("d2 sextile (low $\\rightarrow$ high)")
    ax.set_ylabel("replay probability\n(/ session mean)")
    ax.set_title("pooled counts per d2 sextile", fontsize=9)

    for a in axes:
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    fig.tight_layout()
    assert_local_path(FIGDIR); os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(
            FIGDIR, f"fig2d_binomial_replay{tag}.{ext}")), dpi=200, bbox_inches="tight")
    print(f"\nfigure -> {FIGDIR}\\fig2d_binomial_replay.png/.svg")
    print(f"tables -> {LOGDIR}\\r35_binomial_replay*.csv")


if __name__ == "__main__":
    main()
