"""
r16_robustness_index.py -- indices for the robustness of a predictor's relation to event
type across epochs and recordings.

Two families, both on the 30 min block classifier of `r16_block_level_dissociation.py`
(barrages against SWRs, predictors on event-free time, z-scored within block; blocks of
>= 60 windows with >= MIN_EVENTS of both event types).

A. META-ANALYTIC, from the per-block coefficient and its SE (ORTHO=1 table, where the
   competitors are residualised on one another so their coefficients are not variance-
   inflated; d2's coefficient is identical to the raw fit):
     tau          between-block SD of the TRUE effect (DerSimonian-Laird), i.e. how much
                  the relation itself moves from epoch to epoch beyond sampling error;
     I^2          share of the variance of the block estimates that is heterogeneity;
     mu / tau     the pooled effect in units of its own between-epoch spread -- a
                  standardised "consistency" (Cohen's d over epochs);
     Phi(|mu|/tau)  the fraction of epochs expected to share the sign of the pooled
                  effect (checked against the observed fraction);
     95 % PI      prediction interval for the effect in a NEW epoch,
                  mu +- 1.96 sqrt(tau^2 + SE_mu^2) (IntHout et al. 2016) -- the
                  meta-analytic statement of "will it replicate".

B. OPERATIONAL, leave-one-session-out TRANSFER: the slope is learned on every other
   session's blocks (one binomial GLM with a per-block intercept and a common slope) and
   applied to each block of the held-out session with only that block's intercept refit.
   Held-out deviance explained (relative to the block's intercept-only model) can be
   negative when the borrowed slope points the wrong way. Reported against the block's
   own in-sample deviance explained:
     transfer ratio = held-out / in-sample deviance explained  (medians over blocks)
   A predictor whose relation is the same everywhere has a ratio near 1; one whose
   relation is strong but epoch-specific has a large in-sample fit and a small ratio.
   Inference on the held-out gain: t over session means; d2 vs MUA paired over sessions.

Both families also get a session-level cluster bootstrap (resample sessions with
replacement, blocks travel with their session) for a 95% CI. Zero-trial windows are
skipped in every likelihood (see r16_block_level_dissociation).

    -> data/revision/log/r16_robustness_index.csv        (A and the B summary)
    -> data/revision/log/r16_robustness_index_loso.csv   (B, per block)
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
import statsmodels.api as sm                                          # noqa: E402
from scipy import stats                                               # noqa: E402

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                        # noqa: E402
from r16_block_level_dissociation import load, LOGDIR                 # noqa: E402
from r16_predictor_heterogeneity import meta, good_blocks, ORTHO_CSV, LAB_ORTHO  # noqa: E402

MODELS = [(["d2_bg"], "d2 alone"), (["MUA_bg"], "MUA alone"), (["d2_bg", "MUA_bg"], "d2 + MUA"),
          (["d2_bg", "MUA_bg", "fano_bg", "sync_bg", "EI_bg"], "all five")]


def meta_indices(C, preds):
    rows = []
    for c in preds:
        v = C[C.predictor == c]
        b, se = v.coef.to_numpy(), v.se.to_numpy()
        m = meta(b, se)
        mu, tau = m["re_coef"], m["tau"]
        se_mu = abs(mu / m["re_z"])
        pi = 1.96 * np.sqrt(tau ** 2 + se_mu ** 2)
        rows.append(dict(predictor=c, label=LAB_ORTHO[c], n_blocks=int(b.size), mu=mu, tau=tau,
                         I2=m["I2"], mu_over_tau=abs(mu) / tau if tau > 0 else np.inf,
                         phi=float(stats.norm.cdf(abs(mu) / tau)) if tau > 0 else 1.0,
                         obs_same_sign=max((b < 0).sum(), (b > 0).sum()) / b.size,
                         pi_lo=mu - pi, pi_hi=mu + pi, re_z=m["re_z"]))
    return pd.DataFrame(rows)


def loso(W, cols):
    B = [(s, blk, h) for s, blk, h in good_blocks(W, cols)]

    def prep(h):
        Z = ((h[cols] - h[cols].mean()) / (h[cols].std(ddof=0) + 1e-12)).to_numpy(float)
        keep = (h.n_barr + h.n_rip).to_numpy() > 0
        return Z[keep], np.column_stack([h.n_barr, h.n_rip])[keep]

    rows = []
    for ho in sorted({s for s, _, _ in B}):
        tr = [b for b in B if b[0] != ho]
        Zs, Ds, Ys = [], [], []
        for k, (s, blk, h) in enumerate(tr):
            Z, Y = prep(h)
            d = np.zeros((len(Z), len(tr)))
            d[:, k] = 1
            Zs.append(Z), Ds.append(d), Ys.append(Y)
        f = sm.GLM(np.vstack(Ys), np.column_stack([np.vstack(Ds), np.vstack(Zs)]),
                   family=sm.families.Binomial()).fit()
        slope = f.params[len(tr):]
        for s, blk, h in [b for b in B if b[0] == ho]:
            Z, Y = prep(h)
            one = np.ones((len(Z), 1))
            f0 = sm.GLM(Y, one, family=sm.families.Binomial()).fit()
            f1 = sm.GLM(Y, one, family=sm.families.Binomial(), offset=Z @ slope).fit()
            fi = sm.GLM(Y, np.column_stack([one, Z]), family=sm.families.Binomial()).fit()
            rows.append(dict(session=s, block=blk, n_win=len(h),
                             heldout=100 * (f0.deviance - f1.deviance) / f0.deviance,
                             insample=100 * (f0.deviance - fi.deviance) / f0.deviance,
                             **{f"slope_{c}": float(v) for c, v in zip(cols, slope)}))
    return pd.DataFrame(rows)


NBOOT = int(os.environ.get("NBOOT", "2000"))


def cluster_boot(sessions, fn, nboot=NBOOT, seed=0):
    """Session-level cluster bootstrap: resample sessions with replacement, hand the list
    of drawn sessions (with repeats) to fn, return the percentile 95 % CI of its value."""
    rng = np.random.default_rng(seed)
    sessions = np.asarray(sessions)
    vals = []
    for _ in range(nboot):
        v = fn(rng.choice(sessions, size=sessions.size, replace=True))
        if v is not None and np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), int(nboot - vals.size), vals


def boot_A(A, C):
    """95 % CI on |mu|/tau per predictor, resampling sessions (blocks travel with them)."""
    lo, hi, nbad, reps = [], [], [], {}
    for c in A.predictor:
        v = C[C.predictor == c]
        by = {s: (g.coef.to_numpy(), g.se.to_numpy()) for s, g in v.groupby("session")}

        def ratio(drawn, by=by):
            bb = np.concatenate([by[s][0] for s in drawn])
            ss = np.concatenate([by[s][1] for s in drawn])
            m = meta(bb, ss)
            return abs(m["re_coef"]) / m["tau"] if m["tau"] > 0 else np.inf

        l, h, nb, vals = cluster_boot(list(by), ratio)
        lo.append(l), hi.append(h), nbad.append(nb)
        reps[c] = vals
    return A.assign(mu_over_tau_lo=lo, mu_over_tau_hi=hi, boot_dropped=nbad), reps


def boot_B(B, per):
    """95 % CI on the transfer ratio per model, resampling sessions."""
    lo, hi, nbad, reps = [], [], [], {}
    for name in B.model:
        R = per[name]
        by = {s: g[["heldout", "insample"]].to_numpy() for s, g in R.groupby("session")}

        def ratio(drawn, by=by):
            X = np.vstack([by[s] for s in drawn])
            ins = np.median(X[:, 1])
            return np.median(X[:, 0]) / ins if ins > 0 else None

        l, h, nb, vals = cluster_boot(list(by), ratio)
        lo.append(l), hi.append(h), nbad.append(nb)
        reps[name] = vals
    return B.assign(transfer_ratio_lo=lo, transfer_ratio_hi=hi, boot_dropped=nbad), reps


def plot(A, B, VA, VB):
    """Two box plots. Each box is the
    session-level cluster-bootstrap distribution of the index (B = NBOOT resamples of the
    sessions, blocks travelling with them): box = interquartile range, whiskers =
    2.5th-97.5th percentile (the 95 % CI), marker = the point estimate on the full data.
    A: |mu|/tau per predictor, Phi on the right-hand axis. B: transfer ratio per model."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from r16_block_level_dissociation import _BASECOLOR, FIGDIR
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                                             "DejaVu Sans"],
                         "font.size": 11, "axes.linewidth": 1.4,
                         "xtick.major.width": 1.4, "ytick.major.width": 1.4})
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.9),
                                 gridspec_kw=dict(width_ratios=[1.25, 1.0]))

    def boxes(axis, data, colors, est, labels, bold):
        bp = axis.boxplot(data, positions=range(len(data)), widths=0.5, whis=(2.5, 97.5),
                          showfliers=False, patch_artist=True, zorder=2,
                          medianprops=dict(color="k", lw=1.6),
                          whiskerprops=dict(lw=1.4), capprops=dict(lw=1.4),
                          boxprops=dict(lw=1.4))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.75)
        for i, (v, c) in enumerate(zip(est, colors)):
            axis.plot(i, v, "D", ms=7, color="white", mec="k", mew=1.4, zorder=4)
            small = labels is not None and v < 0.3      # keep the label off the zero line
            axis.text(i + 0.31, v + (0.04 if small else 0.0),
                      f"{v:.2f}" + (labels[i] if labels else ""), ha="left",
                      va="bottom" if small else "center", fontsize=9.5,
                      fontweight="bold" if bold[i] else "normal")

    # ---- A ----
    colsA = [_BASECOLOR[c[:-3]] for c in A.predictor]
    boxes(ax, [np.clip(VA[c], 0, 4.5) for c in A.predictor], colsA, A.mu_over_tau.to_numpy(),
          [f"\n\u03a6 {v:.2f}" for v in A.phi], [c == "d2_bg" for c in A.predictor])
    ax.axhline(0, lw=1.2, color="k")
    ax.set_xticks(range(len(A)))
    ax.set_xticklabels([LAB_ORTHO[c].replace(" | ", "\n| ") for c in A.predictor], fontsize=9.5)
    ax.set_xlim(-0.6, len(A) - 0.2)
    ax.set_ylim(-0.15, min(4.5, max(A.mu_over_tau_hi) + 0.4))
    ax.set_ylabel("|\u03bc| / \u03c4   (pooled effect \u00f7 its between-epoch SD)")
    sec = ax.secondary_yaxis("right", functions=(stats.norm.cdf, stats.norm.ppf))
    sec.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])
    sec.set_ylabel("\u03a6(|\u03bc|/\u03c4): expected fraction of epochs\nsharing the sign")
    ax.set_title(f"A  how stable is the relation across epochs?\n"
                 f"{int(A.n_blocks.iloc[0])} blocks of 30 min, 11 sessions", loc="left", fontsize=10.5)
    ax.spines["top"].set_visible(False)
    # ---- B ----
    cmap = {"d2 alone": _BASECOLOR["d2"], "MUA alone": _BASECOLOR["MUA"],
            "d2 + MUA": "0.6", "all five": "0.35"}
    bx.axhspan(1.0, 2.0, color="0.93", zorder=0)
    bx.axhline(1.0, ls="--", lw=1.2, color="k", zorder=1)
    bx.axhline(0.0, lw=1.2, color="k", zorder=1)
    bx.text(len(B) - 0.45, 1.03, "transfers whole", ha="right", va="bottom", fontsize=8.5, color="0.35")
    bx.text(-0.55, 0.03, "nothing transfers", ha="left", va="bottom", fontsize=8.5, color="0.35")
    boxes(bx, [VB[m] for m in B.model], [cmap[m] for m in B.model], B.transfer_ratio.to_numpy(),
          None, [m == "d2 alone" for m in B.model])
    bx.set_xticks(range(len(B)))
    bx.set_xticklabels(B.model)
    bx.set_xlim(-0.6, len(B) - 0.3)
    bx.set_ylim(min(-0.2, B.transfer_ratio_lo.min() - 0.1), max(1.3, B.transfer_ratio_hi.max() + 0.15))
    bx.set_ylabel("transfer ratio\n(held-out \u00f7 in-sample deviance explained)")
    bx.set_title("B  does the relation transfer to a new session?\n"
                 "slope learned on the other 10 sessions", loc="left", fontsize=10.5)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    fig.text(0.01, 0.006,
             f"Boxes: distribution of the index over {NBOOT} session-level bootstrap resamples (11 sessions "
             "drawn with replacement, blocks travelling with them); box = IQR, whiskers = 2.5th\u201397.5th "
             "percentile (95 % CI);\ndiamond = the estimate on the full data. A: random-effects pooling of the "
             "per-block coefficients (competitors residualised on one another); \u03bc = pooled effect, "
             "\u03c4 = between-block SD of the true effect;\n|\u03bc|/\u03c4 is a Cohen's d over epochs, \u03a6 its "
             "normal CDF. B: leave-one-session-out; held-out = each block scored with a slope learned on "
             "the other sessions, only its intercept refit.",
             fontsize=8, color="0.3", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    assert_local_path(FIGDIR)
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGDIR, f"r16_robustness_index.{ext}"), dpi=300)
    plt.close(fig)
    print(f"figure -> {os.path.join(FIGDIR, 'r16_robustness_index')}.png/.svg")


def main():
    W = load()
    C = pd.read_csv(ORTHO_CSV)
    preds = ["d2_bg", "MUA_bg", "EI_bg", "fano_bg", "sync_bg"]
    A = meta_indices(C, preds)
    print(f"A. Meta-analytic robustness of the block coefficient ({int(A.n_blocks.iloc[0])} blocks, ORTHO table)")
    print(f"   {'predictor':28s}{'mu':>8s}{'tau':>7s}{'I2':>5s}{'mu/tau':>8s}{'Phi':>6s}{'observed':>10s}"
          f"{'95% PI for a new epoch':>24s}{'RE z':>7s}")
    for _, r in A.iterrows():
        print(f"   {r.label:28s}{r.mu:+8.3f}{r.tau:7.3f}{r.I2:5.0f}{r.mu_over_tau:8.2f}{r.phi:6.2f}"
              f"{r.obs_same_sign:10.2f}     [{r.pi_lo:+.2f}, {r.pi_hi:+.2f}]{r.re_z:+7.2f}")

    print("\nB. Leave-one-session-out transfer of the slope (held-out block: intercept refit, slope borrowed)")
    print(f"   {'model':10s}{'in-sample med':>14s}{'held-out med':>13s}{'held-out mean':>14s}{'blocks>0':>10s}"
          f"{'sess>0':>8s}{'t, p (sessions)':>18s}{'transfer ratio':>16s}")
    rows, per = [], {}
    for cols, name in MODELS:
        R = loso(W, cols)
        per[name] = R
        gm = R.groupby("session")[["heldout", "insample"]].mean()
        t, p = stats.ttest_1samp(gm.heldout, 0.0)
        ratio = R.heldout.median() / R.insample.median()
        print(f"   {name:10s}{R.insample.median():14.2f}{R.heldout.median():13.2f}{R.heldout.mean():14.2f}"
              f"{int((R.heldout > 0).sum()):6d}/{len(R):<3d}{int((gm.heldout > 0).sum()):5d}/{len(gm):<2d}"
              f"{t:+8.2f}, {p:.4f}{ratio:16.2f}")
        rows.append(dict(model=name, n_blocks=len(R), n_sessions=len(gm),
                         insample_median=R.insample.median(), heldout_median=R.heldout.median(),
                         heldout_mean=R.heldout.mean(), blocks_heldout_pos=int((R.heldout > 0).sum()),
                         sessions_heldout_pos=int((gm.heldout > 0).sum()), t=float(t), p=float(p),
                         transfer_ratio=ratio,
                         transfer_ratio_sessions=gm.heldout.mean() / gm.insample.mean()))
    M = per["d2 alone"].merge(per["MUA alone"], on=["session", "block"], suffixes=("_d2", "_MUA"))
    gm = M.groupby("session")[["heldout_d2", "heldout_MUA", "insample_d2", "insample_MUA"]].mean()
    d = gm.heldout_d2 - gm.heldout_MUA
    print(f"\n   d2 - MUA held-out, paired over sessions: mean {d.mean():+.2f} pp, d2 better in "
          f"{int((d > 0).sum())}/{len(d)}, signed-rank p = {stats.wilcoxon(d).pvalue:.3f}; "
          f"in-sample: MUA larger in {int((gm.insample_MUA > gm.insample_d2).sum())}/{len(gm)} sessions")
    print("   per session (in-sample d2 | MUA ; held-out d2 | MUA):")
    for s, r in gm.iterrows():
        print(f"     {s:14s} {r.insample_d2:6.2f} | {r.insample_MUA:6.2f}   {r.heldout_d2:+6.2f} | {r.heldout_MUA:+6.2f}")
    A, VA = boot_A(A, C)
    Bdf, VB = boot_B(pd.DataFrame(rows), per)
    print(f"\n   session-bootstrap 95 % CIs (B = {NBOOT}):")
    for _, r in A.iterrows():
        print(f"     |mu|/tau {r.label:28s} {r.mu_over_tau:5.2f}  [{r.mu_over_tau_lo:.2f}, {r.mu_over_tau_hi:.2f}]"
              + (f"  ({r.boot_dropped} resamples with tau = 0 dropped)" if r.boot_dropped else ""))
    for _, r in Bdf.iterrows():
        print(f"     transfer {r.model:10s} {r.transfer_ratio:5.2f}  [{r.transfer_ratio_lo:.2f}, {r.transfer_ratio_hi:.2f}]")
    assert_local_path(LOGDIR)
    A.assign(part="A").to_csv(assert_local_path(os.path.join(LOGDIR, "r16_robustness_index.csv")), index=False)
    Bdf.assign(part="B").to_csv(
        assert_local_path(os.path.join(LOGDIR, "r16_robustness_index.csv")), mode="a", index=False)
    pd.concat([R.assign(model=n) for n, R in per.items()]).to_csv(
        assert_local_path(os.path.join(LOGDIR, "r16_robustness_index_loso.csv")), index=False)
    print(f"\n-> {LOGDIR}\\r16_robustness_index.csv, _loso.csv")
    plot(A, Bdf, VA, VB)


if __name__ == "__main__":
    main()
