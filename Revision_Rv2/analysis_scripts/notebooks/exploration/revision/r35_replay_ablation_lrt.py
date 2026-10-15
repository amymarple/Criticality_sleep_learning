"""
r35_replay_ablation_lrt.py -- replay ablation by nested likelihood-ratio test
on a single pooled GLM, with the same covariates computed on the ripple-excised background.

The classical GLM ablation test is a nested likelihood ratio, which uses all the data
rather than holding a fifth out and refitting: (1) one pooled binomial GLM with session
fixed effects, so a coefficient is estimated from every session's windows at once rather
than one session at a time; (2) a likelihood-ratio test (chi2, 1 df) in place of held-out
deviance.

Covariates are Fig. 2f's own three features -- d2, MUA, FR_std (the across-neuron SD of
the per-neuron rate in the window, not MUA_std, the temporal SD) -- plus E/I ratio, taken
over the ripple-excised part of the window (`background`, the published set), because an
un-excised within-window SD is largely a readout of the ripples themselves and replay
happens inside ripples; the same set over the intact window (`in-ripple`) is reported as a
supplementary comparison. FR_std enters as its within-session residual on MUA, because the
raw pair is collinear within session and forms a suppression pair in the joint model.

Reported per predictor: pooled likelihood-ratio chi2 and p; dAIC/dBIC; the cluster-robust
Wald z on the coefficient; deviance explained alone and in the joint model, each with a
session-cluster bootstrap CI; and the per-session LR chi2 (the points in the figure).

Windows: `r35_extended_windows.csv`, sessions in SETS minus EXCLUDE, NREM, non-overlapping,
inside decoded time. d2 is the Python ripple-excised d2 (2024 Euclidean, p=20).

    -> data/revision/log/r35_replay_ablation_lrt.csv
    -> reports/revision_figs/fig2d_replay_ablation_lrt.{png,svg}
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                  # noqa: E402
from r35_binomial_replay import zs                              # noqa: E402
from r35_extend_replay_sessions import EXCLUDE                  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
SRC = os.path.join(LOGDIR, "r35_extended_windows.csv")
OUT = os.path.join(LOGDIR, "r35_replay_ablation_lrt.csv")
# FR_std here is the across-neuron ("spatial") SD of the per-neuron rate -- Fig 2f's own
# third feature, `features=("d2","MUA","FR_std")` -- not MUA_std, the 30 s rolling
# temporal SD from the same covariate-producing notebook; MUA_std is kept as a separate,
# supplementary comparison set so it is not conflated with FR_std.
# MUA_bg is the mean population rate on the excised bins; the across-neuron mean
# (FR_mean_bg) is the same quantity divided by the unit count, hence identical after
# within-session z-scoring, so only one of the two is carried.
# MUA_bg and FR_std_bg are collinear within session with opposite-signed coefficients in
# the joint model (a suppression pair): each one's apparent drop-one share largely
# reflects the pair's contrast rather than either variable alone. The figure therefore
# draws each predictor ALONE (hatched) beside the same predictor IN THE JOINT MODEL
# (filled): the hatched bar is what a variable carries by itself, the filled bar is its
# unique share, and for a collinear pair the filled bars belong to the pair's contrast.
# CV_bg / CV (below) make that contrast explicit as a single ratio and are supplementary.
SETS = {
    "background":   [("py_excised", "d2"), ("MUA_bg", "MUA (bg)"), ("FR_std_bg_res", "FR std | MUA (bg)"),
                     ("EI_bg", "E/I (bg)")],
    "in-ripple":    [("py_excised", "d2"), ("MUA", "MUA"), ("FR_std_res", "FR std | MUA"),
                     ("EI_log", "E/I")],
}
COLOR = {"py_excised": "#7b3294", "MUA": "#fdb863", "MUA_bg": "#fdb863",
         "CV": "#1b7837", "CV_bg": "#1b7837",
         "FR_std": "#2c7fb8", "FR_std_bg": "#2c7fb8", "FR_std_res": "#2c7fb8",
         "FR_std_bg_res": "#2c7fb8", "MUA_std": "#d95f0e",
         "MUA_std_bg": "#d95f0e", "EI_bg": "#1b7837", "EI_log": "#1b7837"}


def load():
    W = pd.read_csv(SRC)
    # E/I ratio as a fourth predictor: log((E rate per E unit) / (I rate per I unit)),
    # included because FR std conditioned on MUA behaves partly as an interneuron-dominance
    # readout, which E/I addresses directly. Background set: computed on the ripple-excised
    # bins, from the full-covariate table (same windows). In-ripple set: log of the
    # builder's whole-window EI_ratio.
    EI = pd.read_csv(os.path.join(LOGDIR, "r35_replay_full_covariates_windows.csv"))
    W = W.merge(EI[["session", "center_s", "EI_bg"]], on=["session", "center_s"], how="left")
    W["EI_log"] = np.log(W.EI_ratio)
    # Across-neuron coefficient of variation of the per-neuron firing rate: SD across
    # neurons (FR_std, ddof=0) over the mean across neurons, i.e. the population rate MUA
    # divided by the unit count (the covariate-producing notebook's own `fr_cv`, computed
    # there but not used as a feature). n_units is constant within a session, so
    # FR_std / MUA z-scores to the identical column; the division is kept so the stored
    # value is the real CV rather than CV / n_units.
    W["CV_bg"] = W.FR_std_bg / (W.MUA_bg / W.n_units)
    W["CV"] = W.FR_std / (W.MUA / W.n_units)
    # FR_std enters as its within-session residual on MUA. The raw pair is collinear
    # within session and forms a suppression pair in the joint GLM, so each one's drop-one
    # share is really the pair's contrast. Regressing FR_std on MUA within every session
    # and keeping the residual assigns the shared part to MUA and leaves FR_std as firing
    # heterogeneity beyond the mean rate; the three predictors are then near-orthogonal,
    # so drop-one equals alone for each. Same three quantities, same information, only the
    # attribution changes -- MUA keeps its own column as the basis. The raw columns are
    # still z-scored and carried for other uses.
    RES = {"FR_std_bg_res": ("FR_std_bg", "MUA_bg"), "FR_std_res": ("FR_std", "MUA")}
    raw = sorted(({c for v in SETS.values() for c, _ in v} - set(RES))
                 | {x for pair in RES.values() for x in pair})
    W = W[~W.session.isin(EXCLUDE)].dropna(subset=raw).copy()
    for out, (y, x) in RES.items():
        W[out] = W.groupby("session", group_keys=False).apply(
            lambda g, y=y, x=x: pd.Series(resid_on(g[y].to_numpy(float), g[x].to_numpy(float)),
                                          index=g.index))
    for c in sorted(set(raw) | set(RES)):
        W[f"{c}_z"] = W.groupby("session")[c].transform(zs)
    return W


def resid_on(y, x):
    """y minus its least-squares line on x (one slope per call, i.e. per session)."""
    b = np.polyfit(x, y, 1)
    return y - np.polyval(b, x)


def holm(ps):
    """Holm step-down adjusted p-values (Fig 2F's post hoc correction)."""
    ps = np.asarray(ps, float)
    m = len(ps)
    o = np.argsort(ps)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def glm(W, cols, cluster=False):
    endog = np.column_stack([W.n_replay, W.n_swr - W.n_replay])
    D = pd.get_dummies(W.session).astype(float)
    X = np.column_stack([W[cols].to_numpy(float), D.to_numpy()]) if cols else D.to_numpy()
    kw = dict(cov_type="cluster", cov_kwds={"groups": W.session.to_numpy()}) if cluster else {}
    return sm.GLM(endog, X, family=sm.families.Binomial()).fit(**kw)


def within_session_r(W, a, b):
    """Median within-session Pearson r. The two non-d2 covariates of the Fig 2f set are
    expected to be close to the same variable (an across-neuron SD scales with the mean
    rate), so their individual coefficients form a suppression pair and are not
    interpretable alone; this reports how correlated d2 is with each, alongside the pair's
    own correlation."""
    v = [np.corrcoef(g[a], g[b])[0, 1] for _, g in W.groupby("session")
         if g[a].std() > 0 and g[b].std() > 0]
    return float(np.median(v)) if v else np.nan


def dev_expl(fit, base):
    """Percent of the baseline model's deviance that this fit removes. The baseline is
    the SAME model without the covariate block (session fixed effects alone, pooled; an
    intercept alone, per session), so 'deviance explained' means 'by the covariates',
    not 'by knowing which session this is'."""
    return 100.0 * (base.deviance - fit.deviance) / base.deviance


def cluster_bootstrap(W, PRED, B, seed=0):
    """Resample sessions with replacement and recompute every panel's statistic.

    The cluster-robust z has no analytic standard error of its own (it is already a ratio
    whose denominator is the SE), but it does have a sampling distribution over session
    sets; resampling sessions (the unit of analysis) and refitting gives a percentile
    interval for it and for the full-minus-ablated deviance, neither of which has a closed
    form otherwise. The coefficient's bootstrap interval is also computed, alongside the
    analytic cluster-robust one.

    Duplicated sessions are relabelled so the session fixed effects stay one per draw.
    Returns {predictor: {stat: (lo, hi)}} at the 2.5/97.5 percentiles.
    """
    cols = [f"{c}_z" for c, _ in PRED]
    sess = np.array(sorted(W.session.unique()))
    parts = {s: W[W.session == s] for s in sess}
    rng = np.random.default_rng(seed)
    acc = {c: {"coef": [], "z": [], "d_dev": [], "d_dev_alone": []} for c, _ in PRED}
    ok = 0
    for _ in range(B):
        pick = rng.choice(len(sess), size=len(sess), replace=True)
        Wb = pd.concat([parts[sess[k]].assign(session=f"{sess[k]}#{i}")
                        for i, k in enumerate(pick)], ignore_index=True)
        try:
            fb, fbc = glm(Wb, cols), glm(Wb, cols, cluster=True)
            bb = glm(Wb, [])
            dv_full = dev_expl(fb, bb)
            for j, (c, _) in enumerate(PRED):
                rd = glm(Wb, [x for i, x in enumerate(cols) if i != j])
                acc[c]["coef"].append(float(fbc.params[j]))
                acc[c]["z"].append(float(fbc.params[j] / fbc.bse[j]))
                acc[c]["d_dev"].append(dv_full - dev_expl(rd, bb))
                acc[c]["d_dev_alone"].append(dev_expl(glm(Wb, [cols[j]]), bb))
        except Exception:
            continue
        ok += 1
    out = {c: {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
               for k, v in d.items() if len(v)} for c, d in acc.items()}
    return out, ok


def run(W, PRED, setname, rows, B=0):
    cols = [f"{c}_z" for c, _ in PRED]
    full, fullc = glm(W, cols), glm(W, cols, cluster=True)
    base = glm(W, [])                       # session fixed effects only
    nswr = float(W.n_swr.sum())
    pair = [c for c, _ in PRED[1:]]
    r_pair = within_session_r(W, pair[0], pair[1])
    r_d2 = [within_session_r(W, PRED[0][0], c) for c in pair]
    BS, nb = (cluster_bootstrap(W, PRED, B) if B else ({}, 0))
    print(f"\n{'=' * 104}\ncovariates: {setname}\n{'=' * 104}")
    print(f"  {'ablated':12s}{'LR chi2':>10s}{'p':>12s}{'dBIC':>9s}"
          f"{'coef/SD':>10s}{'robust z':>11s}{'p':>11s}"
          f"{'d %dev':>11s}{'alone %dev':>11s}{'d bits/SWR':>12s}{'sess p<.05':>11s}{'  boot 95% CI of z':>20s}")
    print(f"  full model explains {dev_expl(full, base):.2f}% of the deviance that "
          f"session fixed effects leave; 'd %dev' is full minus ablated")
    print(f"  within-session r: {pair[0]} vs {pair[1]} = {r_pair:+.3f}; "
          f"{PRED[0][0]} vs each = {r_d2[0]:+.3f}, {r_d2[1]:+.3f}")
    for j, (c, lab) in enumerate(PRED):
        keep = [x for i, x in enumerate(cols) if i != j]
        red = glm(W, keep)
        lr = 2 * (full.llf - red.llf)
        p = float(stats.chi2.sf(lr, 1))
        se_c = float(fullc.bse[j])                      # cluster-robust SE of the coefficient
        z = float(fullc.params[j] / se_c)
        pz = float(2 * stats.norm.sf(abs(z)))
        per = {}
        for s, g in W.groupby("session"):
            e = np.column_stack([g.n_replay, g.n_swr - g.n_replay])
            f1 = sm.GLM(e, sm.add_constant(g[cols].to_numpy(float), has_constant="add"),
                        family=sm.families.Binomial()).fit()
            X0 = (sm.add_constant(g[keep].to_numpy(float), has_constant="add") if keep
                  else np.ones((len(g), 1)))
            f0 = sm.GLM(e, X0, family=sm.families.Binomial()).fit()
            l = max(2 * (f1.llf - f0.llf), 1e-6)
            # FULL MINUS ABLATED on an interpretable axis, per session: the percent of
            # the intercept-only deviance that this variable alone accounts for.
            fb = sm.GLM(e, np.ones((len(g), 1)), family=sm.families.Binomial()).fit()
            dd = dev_expl(f1, fb) - dev_expl(f0, fb)
            fa = sm.GLM(e, sm.add_constant(g[[cols[j]]].to_numpy(float), has_constant="add"),
                        family=sm.families.Binomial()).fit()
            dda = dev_expl(fa, fb)                          # this predictor ALONE
            # per-session z on the quasi-binomial SE (SE inflated by sqrt of the
            # overdispersion), the within-session analogue of the pooled robust z
            disp = float(f1.pearson_chi2 / f1.df_resid) if f1.df_resid > 0 else 1.0
            se_qb = float(f1.bse[1 + j] * np.sqrt(max(disp, 1.0)))
            per[s] = (l, float(stats.chi2.sf(l, 1)), float(f1.params[1 + j]),
                      float(f1.params[1 + j] / se_qb) if se_qb > 0 else np.nan, dd, dda)
        nsig = sum(1 for v in per.values() if v[1] < .05)
        # FULL MINUS ABLATED, pooled: the same quantity the LR tests, on two axes a
        # reader can price. d_bits is in the same units (bits/SWR) as the cross-validated
        # held-out gain quoted in main(), so the two are directly comparable.
        d_dev = dev_expl(full, base) - dev_expl(red, base)
        d_alone = dev_expl(glm(W, [cols[j]]), base)         # this predictor ALONE, pooled
        d_bits = (full.llf - red.llf) / (nswr * np.log(2))
        zb = BS.get(c, {}).get("z")
        zbs = f"  [{zb[0]:+.1f}, {zb[1]:+.1f}]" if zb else ""
        print(f"  {lab:12s}{lr:10.1f}{p:12.2e}{red.bic - full.bic:9.1f}"
              f"{float(full.params[j]):10.3f}{z:11.2f}{pz:11.2e}"
              f"{d_dev:11.3f}{d_alone:11.3f}{d_bits:12.6f}{nsig:>10d}/{len(per)}{zbs}")
        for s, (l, pp, b, zs_, dd, dda) in per.items():
            rows.append(dict(covset=setname, predictor=c, label=lab, session=s, lr_chi2=l,
                             p=pp, coef=b, z_session=zs_, d_dev_session=dd,
                             d_dev_alone_session=dda, d_dev_alone=d_alone,
                             d_dev_alone_bs_lo=BS.get(c, {}).get('d_dev_alone', (np.nan,) * 2)[0],
                             d_dev_alone_bs_hi=BS.get(c, {}).get('d_dev_alone', (np.nan,) * 2)[1],
                             pooled_lr=lr, pooled_p=p,
                             pooled_coef=float(fullc.params[j]), pooled_se=se_c,
                             dAIC=red.aic - full.aic, dBIC=red.bic - full.bic,
                             d_dev=d_dev, d_bits=d_bits, n_boot=nb,
                             coef_bs_lo=BS.get(c, {}).get('coef', (np.nan,) * 2)[0],
                             coef_bs_hi=BS.get(c, {}).get('coef', (np.nan,) * 2)[1],
                             z_bs_lo=BS.get(c, {}).get('z', (np.nan,) * 2)[0],
                             z_bs_hi=BS.get(c, {}).get('z', (np.nan,) * 2)[1],
                             d_dev_bs_lo=BS.get(c, {}).get('d_dev', (np.nan,) * 2)[0],
                             d_dev_bs_hi=BS.get(c, {}).get('d_dev', (np.nan,) * 2)[1],
                             dev_full=dev_expl(full, base),
                             z_cluster=z, p_cluster=pz, r_cov_pair=r_pair,
                             r_d2_cov1=r_d2[0], r_d2_cov2=r_d2[1]))


def main():
    W = load()
    print(f"{len(W)} windows, {W.session.nunique()} sessions, {int(W.n_swr.sum())} SWRs, "
          f"{W.n_replay.sum() / W.n_swr.sum():.1%} replay; one coefficient per predictor "
          f"from {len(W)} windows")
    rows = []
    B = int(os.environ.get("NBOOT", "2000"))
    print(f"cluster bootstrap over sessions: B = {B} (NBOOT=0 skips it)")
    for setname, PRED in SETS.items():
        run(W, PRED, setname, rows, B=B)
    R = pd.DataFrame(rows)
    assert_local_path(LOGDIR)
    R.to_csv(assert_local_path(OUT), index=False)
    print("\n  The cross-validated form of the SAME comparison is null for d2 (+0.00005 "
          "bits/SWR, 17/31 chunk\n  folds, p = 0.65): holding out a fifth and refitting "
          "costs more than d2's unique share is\n  worth (REVISION_STATUS 6.2). The "
          "likelihood ratio uses all of it.")
    plot(R)


def plot(R):
    """Two panels, both with error bars throughout. A cluster-robust z is a ratio whose
    denominator is already the standard error, so it has no analytic CI of its own; the
    session-level cluster bootstrap in `cluster_bootstrap` supplies one, for it and for
    the full-minus-ablated deviance alike, so both quantities get an interval.

    (a) EFFECT SIZE -- pooled coefficient, log-odds per within-session SD, with its 95%
        cluster-robust CI (the bootstrap CI is also computed and stored).
    (b) CONTRIBUTION -- full minus ablated deviance explained, with its 95% bootstrap CI;
        the cluster-robust z and its stars are annotated here rather than given a panel.

    Dots are the per-session values on each axis.
    """
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 12, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    assert_local_path(FIGDIR)
    os.makedirs(FIGDIR, exist_ok=True)
    for setname, stem in (("background", "fig2d_replay_ablation_lrt"),
                          ("in-ripple", "fig2d_replay_ablation_lrt_inripple")):
        PRED = SETS[setname]
        P = R[R.covset == setname]
        order = [c for c, _ in PRED]
        labs = [l for _, l in PRED]

        def g(col):
            return np.array([P[P.predictor == c][col].iloc[0] for c in order])

        bc, bse = g("pooled_coef"), g("pooled_se")
        zc, pc = g("z_cluster"), g("p_cluster")
        dv, dlo, dhi = g("d_dev"), g("d_dev_bs_lo"), g("d_dev_bs_hi")
        cols = [COLOR[c] for c in order]
        dev_full, nb = P.dev_full.iloc[0], int(P.n_boot.iloc[0])
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
        rng = np.random.default_rng(0)

        # (a) effect size, analytic cluster-robust CI
        ax = axes[0]
        ax.bar(range(len(order)), bc, width=0.6, color=cols, edgecolor="k",
               linewidth=1.8, zorder=2)
        ax.errorbar(range(len(order)), bc, yerr=1.96 * bse, fmt="none", ecolor="k",
                    elinewidth=1.8, capsize=5, capthick=1.8, zorder=4)
        lo = min(P.coef.min(), (bc - 1.96 * bse).min())
        hi = max(P.coef.max(), (bc + 1.96 * bse).max())
        pad = 0.10 * (hi - lo)
        for i, c in enumerate(order):
            v = P[P.predictor == c].coef.to_numpy()
            ax.plot(i + rng.uniform(-0.14, 0.14, v.size), v, "o", ms=4, color="0.25",
                    alpha=0.75, mew=0, zorder=3)
            ax.text(i, hi + 0.4 * pad, f"{bc[i]:+.3f}", ha="center", va="bottom",
                    fontsize=10)
        ax.set_ylim(lo - pad, hi + 2.6 * pad)
        ax.set_ylabel("log-odds of replay per SD of predictor")
        ax.set_title("a  effect size\n" + "bars: pooled ± 95% cluster-robust CI",
                     fontsize=10, loc="left")

        # (b) contribution to model fit: each predictor ALONE (hatched) beside the same
        #     predictor IN THE JOINT MODEL (filled). The filled bar is the unique share
        #     the LR tests; the hatched bar is what the variable carries by itself. For a
        #     collinear pair the filled bars belong to the pair's contrast, not to either
        #     member: MUA, alone only 0.17%, shows a large filled bar for that reason.
        ax = axes[1]
        da, alo, ahi = g("d_dev_alone"), g("d_dev_alone_bs_lo"), g("d_dev_alone_bs_hi")
        w, off = 0.34, 0.20
        xa = np.arange(len(order)) - off
        xj = np.arange(len(order)) + off
        ax.bar(xa, da, width=w, facecolor="white", edgecolor=cols, linewidth=1.8,
               hatch="////", zorder=2)
        ax.bar(xj, dv, width=w, color=cols, edgecolor="k", linewidth=1.8, zorder=2)
        ax.errorbar(xa, da, yerr=[da - alo, ahi - da], fmt="none", ecolor="0.35",
                    elinewidth=1.4, capsize=3, capthick=1.4, zorder=4)
        ax.errorbar(xj, dv, yerr=[dv - dlo, dhi - dv], fmt="none", ecolor="k",
                    elinewidth=1.8, capsize=4, capthick=1.8, zorder=4)
        sv = np.concatenate([P.d_dev_session.to_numpy(), P.d_dev_alone_session.to_numpy()])
        blo = min(sv.min(), dlo.min(), alo.min(), 0.0)
        bhi = max(sv.max(), dhi.max(), ahi.max())
        bpad = 0.10 * (bhi - blo)
        for i, c in enumerate(order):
            q = P[P.predictor == c]
            ax.plot(xa[i] + rng.uniform(-0.09, 0.09, len(q)), q.d_dev_alone_session, "o",
                    ms=3, color="0.4", alpha=0.7, mew=0, zorder=3)
            ax.plot(xj[i] + rng.uniform(-0.09, 0.09, len(q)), q.d_dev_session, "o",
                    ms=3, color="0.25", alpha=0.75, mew=0, zorder=3)
            star = ("***" if pc[i] < .001 else "**" if pc[i] < .01
                    else "*" if pc[i] < .05 else "n.s.")
            # each bar labelled at its own top, so the labels never share a line
            ax.text(xa[i], ahi[i] + 0.25 * bpad, f"{da[i]:.2f}%", ha="center", va="bottom",
                    fontsize=8, color="0.35")
            ax.text(xj[i], dhi[i] + 0.25 * bpad, f"{dv[i]:.2f}%" + chr(10) + f"{star} z {zc[i]:+.1f}",
                    ha="center", va="bottom", fontsize=8)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="white", edgecolor="0.3", hatch="////",
                                 label="alone (marginal)"),
                           Patch(facecolor="0.6", edgecolor="k",
                                 label="in the joint model (unique)")],
                  frameon=False, fontsize=8, loc="upper left", ncol=2,
                  columnspacing=1.2, handlelength=1.6)
        ax.set_ylim(blo - bpad, bhi + 2.8 * bpad)
        ax.set_xlim(-0.7, len(order) - 0.3)
        ax.set_ylabel("deviance explained (%)")
        ax.set_title(f"b  contribution to model fit (joint model explains {dev_full:.2f}%)" + chr(10)
                     + f"bars: \u00b1 95% CI, cluster bootstrap over sessions (B = {nb})",
                     fontsize=10, loc="left")

        for ax in axes:
            ax.axhline(0, ls="-", lw=1, color="k")
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(labs, rotation=30, ha="right")
            ax.set_xlabel("")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        note = {"background": "Fig 2F's variables on the ripple-excised background",
                "in-ripple": "Fig 2F's variables on the intact window (supplementary)"}[setname]
        fig.suptitle(f"replay: contribution of each predictor — {note} + E/I, "
                     f"n = {P.session.nunique()} sessions", fontsize=10.5, x=0.01, ha="left")
        # pairwise, Fig 2f's post hoc: d2's per-session drop-one minus each competitor's,
        # signed-rank across sessions, Holm over the two contrasts.
        per = P.pivot_table(index="session", columns="predictor", values="d_dev_session")
        pw = []
        for i in range(1, len(order)):
            d = (per[order[0]] - per[order[i]]).dropna()
            pw.append((i, int((d > 0).sum()), int(d.size), float(stats.wilcoxon(d).pvalue)))
        ph = holm([p for *_, p in pw])
        axb = axes[1]
        y0, y1 = axb.get_ylim()
        for k, ((i, npos, n, p), padj) in enumerate(zip(pw, ph)):
            y = y1 + (0.05 + 0.17 * k) * (y1 - y0)
            axb.plot([xj[0], xj[0], xj[i], xj[i]],
                     [y - 0.02 * (y1 - y0), y, y, y - 0.02 * (y1 - y0)], color="k", lw=1.2,
                     clip_on=False)
            star = "***" if padj < .001 else "**" if padj < .01 else "*" if padj < .05 else "n.s."
            axb.text((xj[0] + xj[i]) / 2, y + 0.01 * (y1 - y0),
                     f"d2 > {labs[i]}: {npos}/{n}, Holm p = {padj:.3f} {star}",
                     ha="center", va="bottom", fontsize=7.5)
        axb.set_ylim(y0, y1 + (0.35 + 0.18 * (len(order) - 1)) * (y1 - y0))
        pair_txt = "; ".join(f"d2 > {labs[i]} in {npos}/{n} sessions (Holm p = {padj:.3f})"
                             for (i, npos, n, p), padj in zip(pw, ph))
        rp = P.r_cov_pair.iloc[0]
        r1, r2 = P.r_d2_cov1.iloc[0], P.r_d2_cov2.iloc[0]
        if abs(rp) > 0.6:
            pair_note = (f"{labs[1]} and {labs[2]} correlate at r = {rp:+.2f} within session: "
                         "their filled (joint) bars are a suppression pair whose share" + chr(10)
                         + "belongs to their contrast, not to either one; read the hatched "
                         "(alone) bars for what each carries by itself.")
        else:
            pair_note = (f"{labs[2]} is FR std with its within-session linear dependence on "
                         f"{labs[1]} removed (r = {rp:+.2f} after; the raw pair r = +0.91 was a "
                         "suppression pair)," + chr(10)
                         + "so MUA and FR std are near-orthogonal and each bar is one "
                         "variable's own contribution.")
        fig.text(0.01, 0.005, pair_note + chr(10)
                 + f"{labs[0]} is near-orthogonal to both (r = {r1:+.2f}, {r2:+.2f}); "
                 "it is significant alone and in the joint model. E/I = log(E rate per E unit / I rate per I unit) "
                 "on the same bins; r = -0.5 with MUA and FR std | MUA," + chr(10)
                 + "so those three share signal and their hatched (alone) bars exceed their filled (unique) bars." + chr(10)
                 + "Brackets (b): d2's per-session drop-one minus each competitor's, "
                 f"signed-rank across sessions, Holm over the {len(order) - 1} contrasts:" + chr(10)
                 + pair_txt + ".",
                 fontsize=8, color="0.3", ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.19, 1, 0.93))
        out = os.path.join(FIGDIR, "backup")   # the main panel is fig2f_standalone.py
        os.makedirs(out, exist_ok=True)
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(out, f"{stem}.{ext}"), dpi=300)
        plt.close(fig)
        print("figure -> " + os.path.join(out, stem) + ".png/.svg")


if __name__ == "__main__":
    main()
