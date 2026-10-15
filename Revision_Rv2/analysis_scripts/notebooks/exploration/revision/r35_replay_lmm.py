"""
r35_replay_lmm.py -- replay vs the four background predictors as a linear mixed
model, windows as the observations, session as the unit, animal above session.

A mixed model uses every window while charging the test for between-session
heterogeneity: each predictor gets a random slope per session, so the fixed effect is
tested against how much its slope varies across sessions, not against window noise alone.

    y ~ C(animal) + d2 + MUA + FR std | MUA + E/I      fixed
        + (1 | session) + (0 + x_k | session)          random intercept, independent random slopes

rate target      y = log(n_replay + 0.5)                       (n_replay per 30 s window)
fraction target  y = log((n_replay + 0.5) / (n_swr - n_replay + 0.5))   (empirical logit)

Gaussian LMM by REML (statsmodels MixedLM, exact). Animal is a fixed effect because the
six animals (three with a single session) do not identify a variance component.

Per target: fixed effects with SE, z and p; random-slope SDs; the session-specific slopes
(fixed + BLUP); Wald contrasts b_d2 - b_k from the fixed-effect covariance, Holm over the
three; and the same model without random slopes, to show what the slopes cost. Predictors
z-scored within session (`r35_replay_ablation_lrt.load`).

Figure: panel a is the full-minus-ablated deviance explained of the pooled Poisson GLM for
replay rate (read from r35_replay_rate_model.csv, so one producer owns that number), with
this mixed model's fixed-effect z as the stars and its Wald contrasts as the brackets;
panel b is the per-session drop-one z-scored within session across the four predictors,
with signed-rank stars and brackets. The fraction target is reported only in the CSV.

    -> data/revision/log/r35_replay_lmm.csv
    -> reports/revision_figs/fig2d_replay_lmm.{png,svg}
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
import statsmodels.api as sm                                          # noqa: E402
from scipy import stats                                               # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                        # noqa: E402
from r35_replay_ablation_lrt import load, holm, SETS, COLOR, LOGDIR, FIGDIR  # noqa: E402

OUT = os.path.join(LOGDIR, "r35_replay_lmm.csv")
PRED = SETS["background"]
X = [f"{c}_z" for c, _ in PRED]
LAB = dict(zip(X, [l for _, l in PRED]))
TARGETS = [("rate", "y_rate", "replay RATE: log(n_replay + 0.5)", "effect on log replay rate per within-session SD"),
           ("fraction", "y_frac", "replay FRACTION: empirical logit", "effect on logit(replay fraction) per SD")]


def prepare():
    W = load().reset_index(drop=True)
    W["animal"] = W.session.str.split("_").str[0]
    W["y_rate"] = np.log(W.n_replay + 0.5)
    W["y_frac"] = np.log((W.n_replay + 0.5) / (W.n_swr - W.n_replay + 0.5))
    return W


def fit_lmm(W, y, slopes):
    fe = " + ".join(X)
    vcf = {f"s_{k}": f"0 + {k}" for k in X} if slopes else None
    m = sm.MixedLM.from_formula(f"{y} ~ C(animal) + {fe}", groups="session", re_formula="1",
                                vc_formula=vcf, data=W)
    r = m.fit(reml=True, method=["lbfgs", "powell"], maxiter=5000)
    assert r.converged, f"LMM did not converge ({y}, slopes={slopes})"
    return r


def session_slopes(r, k):
    """fixed effect + the session's BLUP for predictor k (0 without random slopes)."""
    out = {}
    for s, re in r.random_effects.items():
        b = [v for n, v in re.items() if n.startswith(f"s_{k}")]
        out[s] = float(r.params[k] + (b[0] if b else 0.0))
    return out


def star(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def analyse(W, rows):
    res = {}
    for target, y, title, _ in TARGETS:
        for slopes in (True, False):
            r = fit_lmm(W, y, slopes)
            tag = "random intercept + slopes" if slopes else "random intercept only"
            print(f"\n{title} | session {tag}, animal fixed | {len(W)} windows, {W.session.nunique()} sessions")
            print(f"  {'fixed effect':22s}{'coef':>8s}{'se':>8s}{'z':>7s}{'p':>10s}"
                  + ("   slope SD" if slopes else ""))
            vs = dict(zip(X, np.sqrt(np.asarray(r.vcomp, float)))) if slopes else {}
            # without random slopes every session shares the fixed slope (and the intercept
            # variance is 0 once animal is fixed, so BLUPs are not even defined)
            per = ({k: session_slopes(r, k) for k in X} if slopes else
                   {k: {s: float(r.params[k]) for s in W.session.unique()} for k in X})
            for k in X:
                print(f"  {LAB[k]:22s}{r.params[k]:+8.3f}{r.bse[k]:8.3f}{r.tvalues[k]:7.1f}{r.pvalues[k]:10.2e}"
                      + (f"   {vs[k]:.3f}" if slopes else ""))
                rows.append(dict(target=target, model=tag, predictor=k, label=LAB[k], coef=float(r.params[k]),
                                 se=float(r.bse[k]), z=float(r.tvalues[k]), p=float(r.pvalues[k]),
                                 slope_sd=vs.get(k, np.nan),
                                 session_intercept_sd=float(np.sqrt(r.cov_re.iloc[0, 0])),
                                 residual_sd=float(np.sqrt(r.scale))))
                for s, v in per[k].items():
                    rows.append(dict(target=target, model=tag, predictor=k, label=LAB[k], session=s, session_slope=v))
            C = r.cov_params().loc[X, X]
            con = []
            for k in X[1:]:
                e = float(r.params[X[0]] - r.params[k])
                se = float(np.sqrt(C.loc[X[0], X[0]] + C.loc[k, k] - 2 * C.loc[X[0], k]))
                z = e / se
                con.append((k, e, se, z, float(2 * stats.norm.sf(abs(z)))))
            ph = holm([p for *_, p in con])
            for (k, e, se, z, p), padj in zip(con, ph):
                npos = int(sum(per[X[0]][s] > per[k][s] for s in per[k]))
                print(f"      b_d2 - b_{LAB[k]:18s} = {e:+.3f} +- {se:.3f}, z = {z:+.2f}, p = {p:.4f}, "
                      f"Holm p = {padj:.4f}; d2's session slope larger in {npos}/{len(per[k])}")
                rows.append(dict(target=target, model=tag, predictor=k, label=f"d2 - {LAB[k]}", contrast=1,
                                 coef=e, se=se, z=z, p=p, p_holm=float(padj), n_pos=npos, n_sessions=len(per[k])))
            if slopes:
                res[target] = dict(r=r, per=per, con=con, ph=ph)
    return res


def plot(res, W):
    """Replay rate only. Panel a: the ablation quantity -- deviance explained lost when
    the predictor is removed (full - ablated, pp) from the pooled Poisson GLM with session
    fixed effects, read from r35_replay_rate_model.csv (cluster-bootstrap CI; dots =
    per-session drop-one); stars above the bars are this mixed model's fixed-effect z,
    brackets its Wald contrasts. Panel b: the same per-session drop-one z-scored within
    session across the four predictors; stars = signed-rank of the z across sessions;
    brackets = d2's z minus the competitor's, signed-rank, Holm."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 12, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    src = os.path.join(LOGDIR, "r35_replay_rate_model.csv")
    assert os.path.exists(src), "run r35_replay_rate_model.py first (pooled drop-one + bootstrap)"
    G = pd.read_csv(src)
    G = G[G.model == "rate"]
    pooled = G[G.session.isna() & G.contrast.isna()].set_index("predictor")
    per = G[G.session.notna()].pivot_table(index="session", columns="predictor", values="drop_one_session")
    raw = [c for c, _ in PRED]
    assert all(c in pooled.index for c in raw) and all(c in per.columns for c in raw), "rate_model CSV predates the E/I set"
    per = per[raw]
    r, con, ph = (res["rate"][k] for k in ("r", "con", "ph"))
    labs = [LAB[k] for k in X]
    cols = [COLOR[c] for c in raw]
    rng = np.random.default_rng(0)
    K = len(raw)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2))

    # (a) full - ablated, pooled, with the mixed model's stars and brackets
    ax = axes[0]
    v = pooled.loc[raw, "drop_one"].to_numpy(float)
    lo, hi = pooled.loc[raw, "bs_lo"].to_numpy(float), pooled.loc[raw, "bs_hi"].to_numpy(float)
    ax.bar(range(K), v, width=0.6, color=cols, edgecolor="k", linewidth=1.8, zorder=2)
    ax.errorbar(range(K), v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", elinewidth=1.8,
                capsize=5, capthick=1.8, zorder=4)
    top = max(hi.max(), np.percentile(per.to_numpy(float), 98))
    for j, (c, k) in enumerate(zip(raw, X)):
        pv = per[c].to_numpy(float)
        ax.plot(j + rng.uniform(-0.13, 0.13, pv.size), np.clip(pv, None, top), "o", ms=4,
                color="0.25", alpha=0.75, mew=0, zorder=3)
        ax.text(j, top * 1.06, f"{v[j]:.2f} pp\nLMM z = {r.tvalues[k]:+.1f} {star(r.pvalues[k])}",
                ha="center", va="bottom", fontsize=8.5)
    for q, ((k, e, se_, z, p), padj) in enumerate(zip(con, ph)):
        j = X.index(k)
        y = top * (1.5 + 0.2 * q)
        ax.plot([0, 0, j, j], [y - 0.03 * top, y, y, y - 0.03 * top], color="k", lw=1.2, clip_on=False)
        ptxt = "< 0.001" if padj < 0.001 else f"= {padj:.3f}"
        ax.text(j / 2, y + 0.02 * top, f"d2 > {LAB[k]}: {chr(916)}{chr(946)} = {e:+.3f}, Holm p {ptxt} {star(padj)}",
                ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(min(0, per.to_numpy(float).min()) - 0.05 * top, top * (1.55 + 0.2 * len(con) + 0.1))
    ax.axhline(0, ls="-", lw=1, color="k")
    ax.set_xticks(range(K)); ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel("deviance explained lost on dropping\n(full " + chr(8722) + " ablated, pp)")
    ax.set_title(f"a  replay RATE, ablation: joint model explains {pooled.loc[raw[0], 'dev_full']:.2f}%\n"
                 "bars: pooled " + chr(177) + " 95% cluster-bootstrap CI; dots: per session", fontsize=10, loc="left")

    # (b) Fig 2F's display: per-session drop-one z-scored within session across predictors
    ax = axes[1]
    Z = per.sub(per.mean(axis=1), axis=0).div(per.std(axis=1, ddof=0) + 1e-12, axis=0)
    zmax = float(np.sqrt(K - 1))
    data = [Z[c].to_numpy(float) for c in raw]
    bp = ax.boxplot(data, positions=range(K), widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color="k", lw=2), whiskerprops=dict(lw=1.8), capprops=dict(lw=1.8),
                    boxprops=dict(lw=1.8), zorder=2)
    for patch, col in zip(bp["boxes"], cols):
        patch.set_facecolor(col); patch.set_alpha(0.85)
    ps = []
    for j, c in enumerate(raw):
        z = Z[c].to_numpy(float)
        ax.plot(j + rng.uniform(-0.13, 0.13, z.size), z, "o", ms=4.5, color="k", alpha=0.5, mew=0, zorder=3)
        p = float(stats.wilcoxon(z).pvalue); ps.append(p)
        ax.text(j, zmax + 0.1, f"top in {int((Z.idxmax(axis=1) == c).sum())}/{len(Z)}\n{star(p)}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold" if j == 0 else "normal")
    pw = []
    for c in raw[1:]:
        d = (Z[raw[0]] - Z[c]).to_numpy(float)
        pw.append((c, int((d > 0).sum()), int(d.size), float(stats.wilcoxon(d).pvalue)))
    phb = holm([p for *_, p in pw])
    y1 = zmax + 1.0
    for q, ((c, npos, n, p), padj) in enumerate(zip(pw, phb)):
        j = raw.index(c)
        y = y1 + 0.32 * q
        ax.plot([0, 0, j, j], [y - 0.05, y, y, y - 0.05], color="k", lw=1.2, clip_on=False)
        ax.text(j / 2, y + 0.03, f"d2 > {LAB[X[j]]}: {npos}/{n} sessions, Holm p = {padj:.3f} {star(padj)}",
                ha="center", va="bottom", fontsize=7.5)
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.set_ylim(-zmax - 0.2, y1 + 0.32 * len(pw) + 0.15)
    ax.set_xticks(range(K)); ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_ylabel(f"drop-one, z-scored within session\nacross the {K} predictors")
    ax.set_title("b  replay RATE, Fig 2F's display: importance z-scored within session\n"
                 f"box + dots: {len(Z)} sessions; stars: signed-rank of z across sessions", fontsize=10, loc="left")
    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("replay rate vs d2, MUA, FR std | MUA and E/I on the ripple-excised background: "
                 f"n = {W.session.nunique()} sessions, {W.animal.nunique()} mice, {len(W)} windows",
                 fontsize=10.5, x=0.01, ha="left")
    fig.text(0.01, 0.005,
             "a: Poisson GLM of n_replay per 30 s window with session fixed effects, predictors z-scored within "
             "session; y = deviance explained by the joint model minus by the model without that predictor "
             "(r35_replay_rate_model.py, B = 1000)." + chr(10)
             + "Stars above bars and the brackets come from the linear mixed model on log(n_replay + 0.5): animal "
             "fixed, session random intercept and a random slope per session for every predictor; stars = "
             "fixed-effect z," + chr(10)
             + "brackets = Wald contrast of the d2 coefficient minus the competitor's, Holm over three. "
             "b: the per-session drop-one of (a) z-scored within each session across the four predictors "
             "(compositional: the four sum to zero," + chr(10)
             + f"|z| {chr(8804)} {zmax:.2f}); brackets = signed-rank across sessions of d2's z minus the competitor's, Holm. "
             "FR std | MUA (bg) = FR std residualised on MUA within session; E/I (bg) = log(E rate per E unit / I rate per I unit).",
             fontsize=8, color="0.3", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 0.93))
    assert_local_path(FIGDIR)
    os.makedirs(os.path.join(FIGDIR, "backup"), exist_ok=True)   # the main panel is fig2f_standalone.py
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGDIR, "backup", f"fig2d_replay_lmm.{ext}"), dpi=300)
    plt.close(fig)
    print(f"\nfigure -> {os.path.join(FIGDIR, 'backup', 'fig2d_replay_lmm')}.png/.svg")


def main():
    W = prepare()
    print(f"{len(W)} windows, {W.session.nunique()} sessions, {W.animal.nunique()} animals; "
          f"predictors: {[LAB[k] for k in X]}")
    rows = []
    res = analyse(W, rows)
    assert_local_path(OUT)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"-> {OUT}")
    plot(res, W)


if __name__ == "__main__":
    main()
