"""
r35_replay_rate_model.py -- replay RATE (replays per window) as the model
target, with the variables of the submitted Fig. 2f, alongside the replay-FRACTION model used elsewhere.

A model of the FRACTION of SWRs that carry replay (binomial, n_replay out of n_swr)
conditions on the ripple count and asks "given these ripples, how many carry replay?";
that is not the same question as replay RATE, and the two rank the predictors differently
because d2 also predicts the SWR count itself. All three variants are run here, on the
same windows and predictors (ripple-excised d2, background MUA, background FR_std
residualised on MUA, background E/I, z-scored within session, session fixed effects):

    RATE            Poisson GLM of n_replay per 30 s window (replay as a count).
    RATE | SWR      the same with log(n_swr) as a covariate -- replay rate with the
                    ripple count allowed for but not fixed.
    FRACTION        binomial n_replay of n_swr, for reference.

Per model: deviance explained by all four predictors together; each predictor alone;
drop-one (deviance lost when that predictor is removed from the full model -- the
quantity plotted and bootstrapped); coefficients with cluster-robust (session) z; and the
per-session signed-rank of the drop-one value and of the d2-minus-competitor difference.
Cluster bootstrap over sessions for the pooled drop-one CIs. Poisson overdispersion is
reported; inference uses the cluster-robust SE, which does not assume equidispersion.

    -> data/revision/log/r35_replay_rate_model.csv
    -> reports/revision_figs/fig2d_replay_rate_model.{png,svg}
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings
from math import factorial

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
from r35_replay_ablation_lrt import load, LOGDIR, FIGDIR, COLOR       # noqa: E402

OUT = os.path.join(LOGDIR, "r35_replay_rate_model.csv")
PRED = [("py_excised", "d2"), ("MUA_bg", "MUA (bg)"), ("FR_std_bg_res", "FR std | MUA (bg)"),
        ("EI_bg", "E/I (bg)")]
COLS = [f"{c}_z" for c, _ in PRED]
NBOOT = int(os.environ.get("NBOOT", "1000"))
MODELS = [("rate", "replay RATE: n_replay per window (Poisson)", []),
          ("rate_swr", "replay RATE | log SWR count as covariate (Poisson)", ["log_nswr"]),
          ("fraction", "replay FRACTION: n_replay of n_swr (binomial)", [])]


def fit(W, cols, model, cluster=False, session_fe=True):
    extra = dict(MODELS)[model] if False else next(m[2] for m in MODELS if m[0] == model)
    allc = list(cols) + extra
    if session_fe:
        D = pd.get_dummies(W.session).astype(float).to_numpy()
        X = np.column_stack([W[allc].to_numpy(float), D]) if allc else D
    else:
        X = sm.add_constant(W[allc].to_numpy(float), has_constant="add") if allc else np.ones((len(W), 1))
    if model == "fraction":
        endog = np.column_stack([W.n_replay, W.n_swr - W.n_replay])
        fam = sm.families.Binomial()
    else:
        endog = W.n_replay.to_numpy(float)
        fam = sm.families.Poisson()
    kw = dict(cov_type="cluster", cov_kwds={"groups": W.session.to_numpy()}) if cluster else {}
    return sm.GLM(endog, X, family=fam).fit(**kw)


def dev_expl(f, base):
    return 100.0 * (base.deviance - f.deviance) / base.deviance


def dropone(devs):
    """Drop-one: deviance lost when each predictor is removed from the full model --
    Fig 2f's ablation, the field's standard contribution measure."""
    tot = devs[frozenset(COLS)]
    return {c: tot - devs[frozenset(x for x in COLS if x != c)] for c in COLS}


def shapley(devs):
    p = len(COLS)
    out = {}
    for c in COLS:
        others = [x for x in COLS if x != c]
        v = 0.0
        for r in range(p):
            for S in itertools.combinations(others, r):
                w = factorial(r) * factorial(p - r - 1) / factorial(p)
                v += w * (devs[frozenset(S) | {c}] - devs[frozenset(S)])
        out[c] = v
    return out


def all_subsets(W, model, session_fe=True):
    base = fit(W, [], model, session_fe=session_fe)
    return {frozenset(S): (dev_expl(fit(W, list(S), model, session_fe=session_fe), base) if S else 0.0)
            for r in range(len(COLS) + 1) for S in itertools.combinations(COLS, r)}


def run_model(W, model, title, rows):
    devs = all_subsets(W, model)
    tot = devs[frozenset(COLS)]
    sh = dropone(devs)
    full = fit(W, COLS, model)
    fullc = fit(W, COLS, model, cluster=True)
    disp = float(full.pearson_chi2 / full.df_resid)
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"  full model: deviance explained {tot:.2f}%; Pearson dispersion {disp:.2f}"
          + ("  (overdispersed; cluster-robust SE used)" if disp > 1.5 else ""))
    # per session: Shapley of each predictor (intercept-only baseline)
    per = {c: {} for c in COLS}
    for s, g in W.groupby("session"):
        try:
            d = all_subsets(g, model, session_fe=False)
        except Exception:
            continue
        shs = dropone(d)
        for c in COLS:
            per[c][s] = shs[c]
    # cluster bootstrap for the pooled Shapley
    sess = np.array(sorted(W.session.unique()))
    parts = {s: W[W.session == s] for s in sess}
    rng = np.random.default_rng(0)
    acc = {c: [] for c in COLS}
    for _ in range(NBOOT):
        pick = rng.choice(len(sess), size=len(sess), replace=True)
        Wb = pd.concat([parts[sess[k]].assign(session=f"{sess[k]}#{i}")
                        for i, k in enumerate(pick)], ignore_index=True)
        try:
            shb = dropone(all_subsets(Wb, model))
        except Exception:
            continue
        for c in COLS:
            acc[c].append(shb[c])
    print(f"  {'predictor':13s}{'coef':>8s}{'robust z':>10s}{'alone':>8s}"
          f"{'drop-one':>9s}{'share':>7s}{'boot 95% CI':>16s}{'per-session':>13s}{'pos':>7s}{'sr p':>8s}")
    for j, (c, lab) in enumerate(PRED):
        col = COLS[j]
        alone = devs[frozenset([col])]
        assert abs(sh[col] - (tot - devs[frozenset(x for x in COLS if x != col)])) < 1e-9
        v = np.array(list(per[col].values()))
        lo, hi = np.percentile(acc[col], [2.5, 97.5]) if acc[col] else (np.nan, np.nan)
        z = float(fullc.params[j] / fullc.bse[j])
        pw = float(stats.wilcoxon(v).pvalue) if v.size >= 5 else np.nan
        print(f"  {lab:13s}{fullc.params[j]:+8.3f}{z:+10.2f}{alone:8.2f}"
              f"{sh[col]:9.2f}{sh[col] / tot:7.2f}{f'[{lo:.2f}, {hi:.2f}]':>16s}"
              f"{np.median(v):13.2f}{int((v > 0).sum()):4d}/{v.size:<3d}{pw:8.4f}")
        rows.append(dict(model=model, predictor=c, label=lab, coef=float(fullc.params[j]),
                         z_cluster=z, alone=alone, drop_one=sh[col],
                         share=sh[col] / tot, bs_lo=lo, bs_hi=hi, dev_full=tot,
                         dispersion=disp, session_median=float(np.median(v)),
                         n_pos=int((v > 0).sum()), n_sessions=int(v.size), p_signrank=pw))
        for s, x in per[col].items():
            rows.append(dict(model=model, predictor=c, label=lab, session=s, drop_one_session=x))
    d2v = per[COLS[0]]
    for j in range(1, len(COLS)):
        ov = per[COLS[j]]
        d = np.array([d2v[s] - ov[s] for s in d2v if s in ov])
        pw = float(stats.wilcoxon(d).pvalue)
        print(f"  d2 - {PRED[j][1]:12s}: d2's drop-one larger in {int((d > 0).sum())}/{d.size} sessions, "
              f"median {np.median(d):+.2f} pp, signed-rank p = {pw:.4f}")
        rows.append(dict(model=model, predictor=PRED[j][0], label=f"d2 - {PRED[j][1]}",
                         contrast=1, n_pos=int((d > 0).sum()), n_sessions=int(d.size),
                         session_median=float(np.median(d)), p_signrank=pw))
    return dict(tot=tot, sh=sh, per=per, acc=acc, disp=disp)


def holm(ps):
    ps = np.asarray(ps, float); m = len(ps); o = np.argsort(ps); adj = np.empty(m); run = 0.0
    for r_, i in enumerate(o):
        run = max(run, (m - r_) * ps[i]); adj[i] = min(1.0, run)
    return adj


def plot(res):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 12, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.0), sharey=False)
    order = [c for c, _ in PRED]
    labs = [l for _, l in PRED]
    cols = [COLOR[c] for c in order]
    rng = np.random.default_rng(0)
    panel = "abc"
    for k, (model, title, _) in enumerate(MODELS):
        ax = axes[k]
        r = res[model]
        v = np.array([r["sh"][COLS[j]] for j in range(len(COLS))])
        lo = np.array([np.percentile(r["acc"][COLS[j]], 2.5) for j in range(len(COLS))])
        hi = np.array([np.percentile(r["acc"][COLS[j]], 97.5) for j in range(len(COLS))])
        ax.bar(range(len(COLS)), v, width=0.6, color=cols, edgecolor="k", linewidth=1.8, zorder=2)
        ax.errorbar(range(len(COLS)), v, yerr=[v - lo, hi - v], fmt="none", ecolor="k",
                    elinewidth=1.8, capsize=5, capthick=1.8, zorder=4)
        allv = np.concatenate([list(r["per"][COLS[j]].values()) for j in range(len(COLS))])
        top = max(hi.max(), np.percentile(allv, 98))
        for j in range(len(COLS)):
            pv = np.array(list(r["per"][COLS[j]].values()))
            ax.plot(j + rng.uniform(-0.13, 0.13, pv.size), np.clip(pv, None, top), "o",
                    ms=4, color="0.25", alpha=0.75, mew=0, zorder=3)
            pw = stats.wilcoxon(pv).pvalue
            star = "***" if pw < .001 else "**" if pw < .01 else "*" if pw < .05 else "n.s."
            ax.text(j, top * 1.06, f"{v[j]:.2f} pp\n"
                    f"{int((pv > 0).sum())}/{pv.size} {star}", ha="center", va="bottom", fontsize=9)
        # pairwise, Fig 2f's post hoc: d2's per-session drop-one minus each competitor's,
        # signed-rank across sessions, Holm over the two contrasts.
        d2v = r["per"][COLS[0]]
        pw = []
        for j in range(1, len(COLS)):
            ov = r["per"][COLS[j]]
            d = np.array([d2v[s_] - ov[s_] for s_ in d2v if s_ in ov])
            pw.append((j, int((d > 0).sum()), int(d.size), float(stats.wilcoxon(d).pvalue)))
        ph = holm([p for *_, p in pw])
        for q, ((j, npos, n, p), padj) in enumerate(zip(pw, ph)):
            y = top * (1.45 + 0.2 * q)
            ax.plot([0, 0, j, j], [y - 0.03 * top, y, y, y - 0.03 * top], color="k", lw=1.2)
            star = "***" if padj < .001 else "**" if padj < .01 else "*" if padj < .05 else "n.s."
            ax.text(j / 2, y + 0.02 * top, f"d2 > {labs[j]}: {npos}/{n}, Holm p = {padj:.3f} {star}",
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_ylim(min(0, allv.min()) - 0.05 * top, top * (1.5 + 0.22 * (len(COLS) - 1)))
        ax.set_xticks(range(len(COLS)))
        ax.set_xticklabels(labs, rotation=30, ha="right")
        ax.axhline(0, ls="-", lw=1, color="k")
        ax.set_title(f"{panel[k]}  {title}\ndrop-one; the joint model explains {r['tot']:.2f}%",
                     fontsize=9.5, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("deviance explained lost on dropping (pp)")
    fig.suptitle("replay, Fig 2F's variables + E/I on the ripple-excised background, " + f"n = {len(res['rate']['per'][COLS[0]])} sessions: "
                 "which target decides who 'wins'", fontsize=10.5, x=0.01, ha="left")
    fig.text(0.01, 0.005,
             "Bars: pooled drop-one deviance lost ± 95% CI, cluster bootstrap over sessions (B = "
             f"{NBOOT}); dots: per session; stars: signed-rank across sessions.\n"
             "On the RATE d2 dominates because d2 also predicts the SWR count (Fig 2); allowing "
             "for the SWR count (b) or conditioning on it (c) moves the question to\n"
             "'given the ripples, how many carry replay', where E/I (bg) is the larger contributor. "
             "Both are true; they are different questions.\n"
             "FR std | MUA (bg) = FR std with its within-session dependence on MUA removed (the raw pair, r = 0.91, "
             "was a suppression pair). E/I (bg) = log(E rate per E unit / I rate per I unit) on the same "
             "ripple-excised bins;\nit correlates at r = -0.5 with MUA (bg) and with FR std | MUA (bg) within "
             "session, so those three share signal and alone > drop-one for them (FR std | MUA alone 2.8% -> 0.75 pp "
             "once E/I is in). d2 is near-orthogonal to all three.",
             fontsize=8, color="0.3", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.15, 1, 0.93))
    assert_local_path(FIGDIR)
    os.makedirs(os.path.join(FIGDIR, "backup"), exist_ok=True)   # the main panel is fig2f_standalone.py
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(FIGDIR, "backup", f"fig2d_replay_rate_model.{ext}"), dpi=300)
    plt.close(fig)
    print(f"\nfigure -> {os.path.join(FIGDIR, 'backup', 'fig2d_replay_rate_model')}.png/.svg")


def main():
    W = load()
    W["log_nswr"] = np.log(W.n_swr.clip(lower=1))
    W["log_nswr"] = W.groupby("session").log_nswr.transform(lambda v: (v - v.mean()) / (v.std(ddof=0) + 1e-12))
    print(f"{len(W)} windows, {W.session.nunique()} sessions, {int(W.n_swr.sum())} SWRs, "
          f"{int(W.n_replay.sum())} replays; drop-one cluster bootstrap B = {NBOOT}")
    rows, res = [], {}
    for model, title, _ in MODELS:
        res[model] = run_model(W, model, title, rows)
    assert_local_path(LOGDIR)
    pd.DataFrame(rows).to_csv(assert_local_path(OUT), index=False)
    print(f"\n-> {OUT}")
    plot(res)


if __name__ == "__main__":
    main()
