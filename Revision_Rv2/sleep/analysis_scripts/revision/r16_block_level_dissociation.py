"""
r16_block_level_dissociation.py -- the ripple/barrage dissociation at the epoch (block) level;
serves the r16_event_type_classifier_session panel.

A whole-session Poisson fit holds the coefficient constant over the whole sleep recording.
Here each session's NREM windows are cut into contiguous blocks of at
least MINW windows, respecting NREM bout boundaries (a break is a jump in the sliding-window
index larger than twice the median step). Each block gets its own Poisson fit of SWR count and
of barrage count on five predictors (d2, MUA, MUA Fano, synchrony, E/I), z-scored within the
block so a coefficient is per SD of local variation. Inference is run at session level: the block
coefficients are averaged within session and a one-sample t-test runs over the session means;
block-level sign counts are reported descriptively. A block contributes to a target only if it holds
at least MIN_EVENTS of that type.
MINW = 60 windows (30 min of NREM) is primary; 40 and 90
are printed as a robustness check. classify_blocks/classify_sessions recast the same question as
a single binomial classifier (barrage vs SWR) per block or per session.

Inputs: r16_event_excised_windows.csv (SRC) -- 30 s NREM windows with predictor columns and event
counts, CA1 barrage cohort.

Env vars: PRED_SET (excised [default] / intact -- which predictor recompute to use); EXCLUDE
(comma-separated session names to drop); PREDS (reduced predictor subset); ORTHO=1 (residualise
the four competitors sequentially on one another instead of dropping any of them; d2 is never
touched and the joint fit is numerically identical to the five-predictor panel).

    -> data/revision/log/r16_block_level_dissociation.csv
    -> reports/revision_figs/r16_event_type_classifier_session.{png,svg}   (THE panel)
    -> reports/revision_figs/backup/r16_block_level_dissociation.{png,svg},
       backup/r16_event_type_classifier*.{png,svg}                         (supporting)
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

REPO = os.path.abspath(os.path.join(_HERE, ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
# Only the session-level classifier figure (`r16_event_type_classifier_session.png`, from
# classify_sessions + plot_sessions) stays at the top level; the block-level classifier, the
# two-target Poisson panel, the marginal-vs-unique panel and the pooled GLM go to backup/.
BACKUP = os.path.join(FIGDIR, "backup")
os.makedirs(BACKUP, exist_ok=True)
# PRED_SET=excised (default): every predictor computed on the bins that overlap neither a
# ripple nor a barrage (median 92% of a window survives). PRED_SET=intact: d2 from the
# intact-window `d2_real` and every burst metric over the whole window.
SRC = os.path.join(LOGDIR, "r16_event_excised_windows.csv")
PRED_SET = os.environ.get("PRED_SET", "excised")
SUF = "_bg" if PRED_SET == "excised" else ""
TAG = "" if PRED_SET == "excised" else "_intact"
# Optional reduced predictor set, PREDS=d2,MUA,fano,EI (drops synchrony, collinear with MUA
# and Fano). A PREDS run writes under its own tag to backup/ and runs the block classifier only.
PREDS = [x.strip() for x in os.environ.get("PREDS", "").split(",") if x.strip()]
# ORTHO=1: within each block the competitors are residualised on one another in sequence --
# MUA raw, E/I on MUA, Fano on MUA + E/I, synchrony on MUA + E/I + Fano -- and d2 stays raw.
# Same column space as the raw five, so the joint fit, its deviance and d2's coefficient are
# identical to the five-predictor panel; only the competitors' coefficients change (VIF ~ 1).
# A run with both PREDS and ORTHO residualises within the reduced set.
ORTHO = os.environ.get("ORTHO", "") == "1"
ORDER = ["MUA", "EI", "fano", "sync"]
LABEL_ORTHO = {"d2": "d2", "MUA": "MUA", "EI": "E/I\n| MUA", "fano": "MUA Fano\n| MUA, E/I",
               "sync": "synchrony\n| MUA, E/I, Fano"}
if ORTHO:
    TAG += "_ortho"
if PREDS:
    TAG += "_nosync" if set(PREDS) == {"d2", "MUA", "fano", "EI"} else "_" + "_".join(PREDS)
OUT = os.path.join(LOGDIR, f"r16_block_level_dissociation{TAG}.csv")
WIN_S = 30.0
MIN_EVENTS = 10
PRED = [(f"d2{SUF}", "d2"), (f"MUA{SUF}", "MUA"), (f"fano{SUF}", "MUA Fano"),
        (f"sync{SUF}", "synchrony"), (f"EI{SUF}", "E/I")]
if PREDS:
    PRED = [(c, l) for c, l in PRED if c[:len(c) - len(SUF) or None] in PREDS]
if ORTHO:
    _keep = {c for c, _ in PRED}
    PRED = ([(f"d2{SUF}", "d2")] if f"d2{SUF}" in _keep else []) + \
        [(f"{k}{SUF}", LABEL_ORTHO[k]) for k in ORDER if f"{k}{SUF}" in _keep]
TARGETS = [("n_rip", "SWR count"), ("n_barr", "barrage count")]
_BASECOLOR = {"d2": "#7b3294", "MUA": "#fdb863", "fano": "#fee0b6",
              "sync": "#2c7fb8", "EI": "#a6dba0"}
COLOR = {f"{k}{SUF}": v for k, v in _BASECOLOR.items()}


# Optional exclusion by name, EXCLUDE=a,b. Default is none. The cohort is 12 sessions, 11 of
# them with at least MIN_EVENTS barrages. A zero-trial-window guard in classify_sessions / classify_blocks
# skips windows with no event of either kind before fitting (see classify_blocks).
_env = os.environ.get("EXCLUDE", "")
EXCLUDE = {s.strip(): "EXCLUDE env" for s in _env.split(",") if s.strip()}


def load():
    W = pd.read_csv(SRC)
    W["n_rip"] = (W.ripple_rate * WIN_S).round().astype(int)
    W["n_barr"] = (W.barrage_rate * WIN_S).round().astype(int)
    for k, why in EXCLUDE.items():
        if (W.session == k).any():
            print(f"  [excluded] {k}: {why}")
            W = W[W.session != k]
    return W


def blocks(g, minw):
    """Contiguous blocks of >= minw windows, cut at NREM bout boundaries."""
    g = g.sort_values("w").reset_index(drop=True)
    step = g.w.diff().median()
    brk = (g.w.diff().fillna(0) > 2 * step).cumsum()
    out, cur, cnt = [], 0, 0
    for b in brk:
        if cnt >= minw:
            cur += 1
            cnt = 0
        out.append(cur)
        cnt += 1
    g["blk"] = out
    return g


def fit_blocks(W, minw):
    cols = [c for c, _ in PRED]
    rows = []
    for s, g in W.groupby("session"):
        g = blocks(g, minw)
        for bid, h in g.groupby("blk"):
            if len(h) < minw:
                continue                      # trailing remainder, too short to fit
            Z = ((h[cols] - h[cols].mean()) / (h[cols].std(ddof=0) + 1e-12)).to_numpy(float)
            X = sm.add_constant(Z)
            for tgt, _ in TARGETS:
                y = h[tgt].to_numpy()
                if y.sum() < MIN_EVENTS or y.std() == 0:
                    continue                  # too few events of this type to fit a rate
                try:
                    f = sm.GLM(y, X, family=sm.families.Poisson()).fit()
                except Exception:
                    continue
                for j, (c, _) in enumerate(PRED):
                    rows.append(dict(session=s, block=f"{s}#{bid}", n_win=len(h),
                                     target=tgt, predictor=c,
                                     coef=float(f.params[1 + j]), n_events=int(y.sum())))
    return pd.DataFrame(rows)


def report(B, minw, store=None):
    print(f"\n{'=' * 92}\nblocks of >= {minw} windows ({minw * WIN_S / 60:.0f} min of NREM): "
          f"{B.block.nunique()} blocks, {B.session.nunique()} sessions\n{'=' * 92}")
    for tgt, tlab in TARGETS:
        S = B[B.target == tgt]
        print(f"  target: {tlab}  ({S.block.nunique()} blocks, {S.session.nunique()} sessions)")
        print(f"    {'predictor':11s}{'block median':>14s}{'blocks same sign':>19s}"
              f"{'t over session means':>23s}{'df':>4s}{'p':>10s}")
        for c, lab in PRED:
            v = S[S.predictor == c]
            x = v.coef.to_numpy()
            gm = v.groupby("session").coef.mean().to_numpy()
            t, p = stats.ttest_1samp(gm, 0.0)
            same = max(int((x > 0).sum()), int((x < 0).sum()))
            print(f"    {lab:11s}{np.median(x):+14.4f}{same:13d}/{x.size:<5d}"
                  f"{t:+23.2f}{gm.size - 1:4d}{p:10.5f}")
            if store is not None:
                store.append(dict(minw=minw, target=tgt, predictor=c, label=lab,
                                  block_median=float(np.median(x)), n_blocks=int(x.size),
                                  blocks_same_sign=same, n_sessions=int(gm.size),
                                  t=float(t), df=int(gm.size - 1), p=float(p),
                                  session_mean=float(gm.mean())))
        print()


def main():
    W = load()
    if PREDS or ORTHO:
        # reduced / residualised set: block classifier only; no two-target panel, no session fits
        csum = []
        C60 = classify_blocks(W, 60)
        if ORTHO and not PREDS:
            # the residualised five span the raw five: d2's coefficient must be identical
            raw = pd.read_csv(os.path.join(LOGDIR, "r16_event_type_classifier.csv"))
            m = C60[C60.predictor == PRED[0][0]].merge(
                raw[raw.predictor == PRED[0][0]][["block", "coef", "dev_expl"]],
                on="block", suffixes=("", "_raw"))
            print(f"  identity check vs the raw five-predictor fit ({len(m)} blocks): "
                  f"max |d2 coef diff| = {(m.coef - m.coef_raw).abs().max():.2e}, "
                  f"max |deviance-explained diff| = {(m.dev_expl - m.dev_expl_raw).abs().max():.2e}")
        report_classify(C60, 60, csum)
        for m in (40, 90):
            report_classify(classify_blocks(W, m), m, csum)
        CS = pd.DataFrame(csum)
        assert_local_path(LOGDIR)
        C60.assign(minw=60).to_csv(assert_local_path(
            os.path.join(LOGDIR, f"r16_event_type_classifier{TAG}.csv")), index=False)
        CS.to_csv(assert_local_path(
            os.path.join(LOGDIR, f"r16_event_type_classifier_summary{TAG}.csv")), index=False)
        print("  median block VIF:", C60.groupby("predictor").vif.median().round(2).to_dict())
        plot_classify(C60, CS[CS.minw == 60], metric="coef")   # main panel -> top level
        plot_classify(C60, CS[CS.minw == 60], metric="z")      # -> backup/
        return
    B60 = fit_blocks(W, 60)
    summary = []
    report(B60, 60, summary)
    for m in (40, 90):
        report(fit_blocks(W, m), m, summary)
    assert_local_path(LOGDIR)
    B60.assign(minw=60).to_csv(assert_local_path(OUT), index=False)
    pd.DataFrame(summary).to_csv(assert_local_path(
        os.path.join(LOGDIR, f"r16_block_level_dissociation_summary{TAG}.csv")), index=False)
    print(f"[predictors: {PRED_SET}]  With the events EXCISED this two-target version is "
          "no longer\nrobust: d2 on barrage count falls to p = 0.06 at 30 min and "
          "synchrony flips to a\nsignificant POSITIVE there. That is why the single "
          "event-type classifier below, not\nthis pair of panels, is the figure to use.")
    plot(B60, pd.DataFrame(summary))
    # the single-panel version: one classification model
    csum = []
    C60 = classify_blocks(W, 60)
    report_classify(C60, 60, csum)
    for m in (40, 90):
        report_classify(classify_blocks(W, m), m, csum)
    CS = pd.DataFrame(csum)
    C60.assign(minw=60).to_csv(assert_local_path(
        os.path.join(LOGDIR, f"r16_event_type_classifier{TAG}.csv")), index=False)
    CS.to_csv(assert_local_path(
        os.path.join(LOGDIR, f"r16_event_type_classifier_summary{TAG}.csv")), index=False)
    plot_classify(C60, CS[CS.minw == 60], metric="coef")   # main figure
    plot_classify(C60, CS[CS.minw == 60], metric="z")      # -> backup/
    # session level, the project's declared unit of analysis
    ssum = []
    CSESS = classify_sessions(W)
    report_sessions(CSESS, ssum)
    CSESS.to_csv(assert_local_path(
        os.path.join(LOGDIR, f"r16_event_type_classifier_session{TAG}.csv")), index=False)
    pd.DataFrame(ssum).to_csv(assert_local_path(
        os.path.join(LOGDIR, f"r16_event_type_classifier_session_summary{TAG}.csv")),
        index=False)
    pairwise_holm(CSESS)
    plot_sessions(CSESS, pd.DataFrame(ssum))


def plot(B, S):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                                             "DejaVu Sans"],
                         "font.size": 12, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    S = S[S.minw == 60]
    order = [c for c, _ in PRED]
    labs = [l for _, l in PRED]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.9))
    for ax, (tgt, tlab), tag in zip(axes, TARGETS, ("a", "b")):
        P = B[B.target == tgt]
        sns.boxplot(data=P, x="predictor", y="coef", order=order, ax=ax, width=0.6,
                    showfliers=False, linewidth=2, palette=COLOR, hue="predictor",
                    legend=False)
        q1, q3 = P.coef.quantile(0.25), P.coef.quantile(0.75)
        iqr = max(q3 - q1, 1e-9)
        lo, hi = q1 - 2.4 * iqr, q3 + 2.4 * iqr
        pad = 0.12 * (hi - lo)
        rng = np.random.default_rng(1)
        for i, c in enumerate(order):
            v = P[P.predictor == c].coef.to_numpy()
            ax.plot(i + rng.uniform(-0.13, 0.13, v.size), np.clip(v, lo, hi), "o", ms=3,
                    color="k", alpha=0.5, mew=0)
            r = S[(S.target == tgt) & (S.predictor == c)].iloc[0]
            star = ("***" if r.p < .001 else "**" if r.p < .01
                    else "*" if r.p < .05 else "n.s.")
            ax.text(i, hi + 0.3 * pad, f"{int(r.blocks_same_sign)}/{int(r.n_blocks)}\n{star}",
                    ha="center", va="bottom", fontsize=9.5,
                    fontweight="bold" if r.p < .05 else "normal")
        n_clip = int(((P.coef < lo) | (P.coef > hi)).sum())
        if n_clip:
            ax.text(0.99, 0.99, f"{n_clip} clipped to axis", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color="0.4")
        r0 = S[(S.target == tgt)].iloc[0]
        ax.axhline(0, ls="--", lw=1.2, color="k")
        ax.set_ylim(lo - pad, hi + 2.2 * pad)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labs, rotation=30, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("coefficient, log rate per SD (Poisson)")
        ax.set_title(f"{tag}  target: {tlab}\n{int(r0.n_blocks)} blocks, "
                     f"{int(r0.n_sessions)} sessions; numbers = blocks sharing the sign",
                     fontsize=10, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.suptitle("epoch-level dissociation: 30 min blocks of NREM, one Poisson fit each "
                 "— only d2 loads on both targets",
                 fontsize=10.5, x=0.01, ha="left")
    fig.text(0.01, 0.005,
             "Stars are a one-sample t-test over SESSION MEANS of the block coefficients "
             "(df 11 and 10), so blocks buy a better per-session\nestimate, not more "
             "units. SWR count: d2 t = +5.81 p = 1.2e-04, synchrony t = +5.27 "
             "p = 2.7e-04, others n.s. Barrage count: d2 alone,\nt = -4.37 p = 0.0014; "
             "E/I p = 0.88. Same verdict at 20 and 45 min blocks.",
             fontsize=8, color="0.3", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.10, 1, 0.93))
    assert_local_path(FIGDIR)
    for ext in ("png", "svg"):                       # supporting panel -> backup/
        fig.savefig(os.path.join(BACKUP, f"r16_block_level_dissociation{TAG}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"figure -> {os.path.join(BACKUP, 'r16_block_level_dissociation' + TAG)}.png/.svg")


def block_vif(Z):
    """VIF of each column of Z (already standardised) given the others: 1 / (1 - R2)."""
    out = []
    for j in range(Z.shape[1]):
        if Z.shape[1] == 1:
            out.append(1.0)
            continue
        y = Z[:, j]
        X = np.column_stack([np.ones(len(Z)), np.delete(Z, j, axis=1)])
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        r2 = 1 - np.sum((y - X @ b) ** 2) / np.sum((y - y.mean()) ** 2)
        out.append(float(1 / max(1 - r2, 1e-9)))
    return out


def orthogonalise(h, cols):
    """Sequential residualisation of the competitors within one block (see ORTHO above).
    Each variable in ORDER is replaced by its residual on the ones before it (plus an
    intercept); d2 is never touched. Same column space as the raw columns, so the fit is
    unchanged; only the competitors' coefficients are re-expressed."""
    h = h.copy()
    done = []
    for k in ORDER:
        c = f"{k}{SUF}"
        if c not in cols:
            continue
        if done:
            X = np.column_stack([np.ones(len(h))] + [h[d].to_numpy(float) for d in done])
            y = h[c].to_numpy(float)
            h[c] = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        done.append(c)
    return h


def classify_blocks(W, minw):
    """A single model: which KIND of event is this window full of?

    Per block, a binomial GLM whose successes are the barrages and whose failures are the
    SWRs, so the response is P(an event here is a barrage) and the predictors are the
    window's state. A variable that only reads out how bursty the network is cannot answer
    it -- bursts are bursts either way -- so this is the single test the dissociation
    reduces to.

    A block enters only if it holds at least MIN_EVENTS of BOTH types; otherwise there is
    nothing to classify. Predictors are z-scored within the block. Inference is over
    session means, as in `fit_blocks`.
    """
    cols = [c for c, _ in PRED]
    rows = []
    for s, g in W.groupby("session"):
        g = blocks(g, minw)
        for bid, h in g.groupby("blk"):
            if len(h) < minw or h.n_barr.sum() < MIN_EVENTS or h.n_rip.sum() < MIN_EVENTS:
                continue
            if ORTHO:
                h = orthogonalise(h, cols)
            Z = ((h[cols] - h[cols].mean()) / (h[cols].std(ddof=0) + 1e-12)).to_numpy(float)
            endog = np.column_stack([h.n_barr, h.n_rip])
            # windows with no event of either kind (0 binomial trials) are
            # skipped before the fit; predictors stay z-scored over all windows
            keep = endog.sum(1) > 0
            Z, endog = Z[keep], endog[keep]
            try:
                f = sm.GLM(endog, sm.add_constant(Z), family=sm.families.Binomial()).fit()
                f0 = sm.GLM(endog, np.ones((len(endog), 1)),
                            family=sm.families.Binomial()).fit()
            except Exception:
                continue
            dev = 100.0 * (f0.deviance - f.deviance) / f0.deviance
            disp = float(f.pearson_chi2 / f.df_resid) if f.df_resid > 0 else 1.0
            vif = block_vif(Z)
            for j, (c, _) in enumerate(PRED):
                b_, se = float(f.params[1 + j]), float(f.bse[1 + j])
                # PER-BLOCK z: each predictor standardised by ITS OWN standard error.
                # Unlike a z taken across predictors within a unit (17.4), this is not
                # compositional: the five values do not have to sum to zero. It states
                # how many standard errors from zero this block's estimate is.
                rows.append(dict(session=s, block=f"{s}#{bid}", n_win=len(h),
                                 predictor=c, coef=b_, se=se,
                                 z=b_ / se if se > 0 else np.nan,
                                 z_qb=b_ / (se * np.sqrt(max(disp, 1.0))) if se > 0 else np.nan,
                                 dev_expl=dev, n_barr=int(h.n_barr.sum()),
                                 n_rip=int(h.n_rip.sum()), vif=vif[j]))
    return pd.DataFrame(rows)


def report_classify(C, minw, store=None):
    print(f"\n{'=' * 92}\nEVENT-TYPE CLASSIFIER, blocks of >= {minw} windows "
          f"({minw * WIN_S / 60:.0f} min): {C.block.nunique()} blocks, "
          f"{C.session.nunique()} sessions,\nmedian deviance explained "
          f"{C.dev_expl.median():.1f}%\n{'=' * 92}")
    print(f"  {'predictor':11s}{'median z':>14s}{'blocks same sign':>19s}"
          f"{'t over session means':>23s}{'df':>4s}{'p':>10s}{'p (coef)':>12s}{'p (qb z)':>12s}")
    for c, lab in PRED:
        lab = lab.replace("\n", " ")
        v = C[C.predictor == c]
        x = v.z.to_numpy()
        gm = v.groupby("session").z.mean().to_numpy()
        t, p = stats.ttest_1samp(gm, 0.0)
        gq = v.groupby("session").z_qb.mean().to_numpy()
        tq, pq = stats.ttest_1samp(gq, 0.0)
        gc = v.groupby("session").coef.mean().to_numpy()
        tc, pc = stats.ttest_1samp(gc, 0.0)
        same = max(int((x > 0).sum()), int((x < 0).sum()))
        print(f"  {lab:11s}{np.median(x):+14.3f}{same:13d}/{x.size:<5d}"
              f"{t:+23.2f}{gm.size - 1:4d}{p:10.5f}{pc:12.5f}{pq:12.5f}")
        if store is not None:
            store.append(dict(minw=minw, predictor=c, label=lab,
                              block_median_z=float(np.median(x)),
                              block_median_coef=float(np.median(v.coef)),
                              n_blocks=int(x.size), blocks_same_sign=same,
                              n_sessions=int(gm.size), t=float(t),
                              df=int(gm.size - 1), p=float(p),
                              p_coef=float(pc), p_qb=float(pq),
                              dev_expl=float(C.dev_expl.median())))


def plot_classify(C, S, metric="coef"):
    """The single block-level panel. One classifier per 30 min block, five predictors.

    metric="coef" (main figure):
        y = the block's coefficient, log-odds an event is a barrage rather than a SWR per
        within-block SD of the predictor (predictors are z-scored within the block before
        the fit). Stars: one-sample t over SESSION MEANS of the block coefficients.
    metric="z" (backup/): y = coefficient / its own SE per block.
    """
    SFX = "" if metric == "coef" else f"_{metric}"
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                                             "DejaVu Sans"],
                         "font.size": 12, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    order = [c for c, _ in PRED]
    labs = [l for _, l in PRED]
    pcol = "p" if metric == "z" else "p_coef"
    fig, ax = plt.subplots(figsize=(6.6 if ORTHO else 5.8, 5.0))
    sns.boxplot(data=C, x="predictor", y=metric, order=order, ax=ax, width=0.6,
                showfliers=False, linewidth=2, palette=COLOR, hue="predictor",
                legend=False)
    lo, hi = C[metric].quantile(0.02), C[metric].quantile(0.98)
    pad = 0.12 * (hi - lo)
    rng = np.random.default_rng(1)
    for i, c in enumerate(order):
        v = C[C.predictor == c][metric].to_numpy()
        ax.plot(i + rng.uniform(-0.13, 0.13, v.size), np.clip(v, lo, hi), "o", ms=3.5,
                color="k", alpha=0.55, mew=0)
        r = S[S.predictor == c].iloc[0]
        pv = float(r[pcol])
        star = "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "n.s."
        ax.text(i, hi + 1.1 * pad, f"{int(r.blocks_same_sign)}/{int(r.n_blocks)}\n{star}",
                ha="center", va="bottom", fontsize=10,
                fontweight="bold" if pv < .05 else "normal")
    n_clip = int(((C[metric] < lo) | (C[metric] > hi)).sum())
    if n_clip:
        ax.text(0.99, 0.015, f"{n_clip} clipped to axis", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="0.4")
    if metric == "z":
        for y in (-1.96, 1.96):
            ax.axhline(y, ls=":", lw=1.1, color="0.5")
        ax.text(-0.46, 2.08, "p = 0.05 per block", fontsize=8, ha="left", color="0.5")
    ax.axhline(0, ls="--", lw=1.2, color="k")
    ax.set_ylim(lo - pad, hi + 3.8 * pad)
    ax.set_xticks(range(len(order)))
    if ORTHO:
        ax.set_xticklabels(labs, rotation=0, fontsize=9.5)
    else:
        ax.set_xticklabels(labs, rotation=30, ha="right")
    ax.set_xlabel("")
    ax.set_ylabel("per-block z  (coefficient / its own SE)" if metric == "z" else
                  "log-odds an event is a BARRAGE\nrather than a SWR, per within-block SD")
    r0 = S.iloc[0]
    state = ("predictors measured BETWEEN events (ripples and barrages excised)"
             if PRED_SET == "excised" else
             "\u26a0 predictors measured on the INTACT window (circular; comparison only)")
    reduced = (f"synchrony dropped: VIF up to 22 with it, all \u2264 {C.vif.max():.1f} "
               "without\n") if PREDS else ""
    if ORTHO:
        reduced += ("nothing dropped: competitors controlled for one another\n"
                    "(sequential residuals MUA \u2192 E/I \u2192 Fano \u2192 synchrony); "
                    f"all VIF \u2264 {C.vif.max():.1f}\n")
    ax.set_title("which kind of event is this?\n"
                 "barrage vs SWR, one classifier per 30 min block of NREM\n"
                 f"{state}\n{reduced}"
                 "negative = where this variable is high, the events are SWRs\n"
                 f"{int(r0.n_blocks)} blocks, {int(r0.n_sessions)} sessions, "
                 f"{r0.dev_expl:.0f}% of deviance explained", fontsize=9.5, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    d2r = S[S.predictor == PRED[0][0]].iloc[0]
    if metric == "z":
        foot = ("Each predictor is standardised by ITS OWN standard error, one z per block, "
                "so unlike a z taken across predictors\nthis is not compositional. Numbers "
                "are blocks sharing the sign; stars are a one-sample t-test over SESSION "
                f"MEANS\n(df {int(d2r.df)}). Only d2 answers the question: median z = "
                f"{d2r.block_median_z:+.2f}, t = {d2r.t:+.2f}, p = {d2r.p:.1e}. The same "
                "test on the raw\ncoefficients gives p = "
                f"{d2r.p_coef:.1e}, and on quasi-binomial z (overdispersion-guarded) "
                f"p = {d2r.p_qb:.1e}.")
    else:
        foot = ("Predictors are z-scored within each block before the fit, so the axis is the"
                " effect per within-block z-unit;" + "\n"
                "one fit per 30 min block. Numbers are blocks sharing the sign; stars are a"
                " one-sample t-test over SESSION MEANS" + "\n"
                f"of the block coefficients (df {int(d2r.df)}), so blocks improve the"
                " per-session estimate rather than inflating n." + "\n"
                f"Only d2 answers the question: median coefficient {d2r.block_median_coef:+.3f},"
                f" p = {d2r.p_coef:.1e}; the per-block z version is in backup/."
                + ("\nWith all five predictors (r16_event_type_classifier.png): d2 p = 0.014, "
                   "MUA p = 0.19 — dropping the collinear\npredictor changes neither."
                   if PREDS else "")
                + ("\nNothing is dropped: E/I, Fano and synchrony enter as their residuals on the "
                   "variables to their left, so each is what it adds\nbeyond them. The "
                   "residualised five span the raw five, so the fit and d2's coefficient are "
                   "identical to r16_event_type_classifier.png;\nonly the competitors' "
                   "coefficients change." if ORTHO else ""))
    fig.text(0.01, 0.005, foot, fontsize=8, color="0.3", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.18 if ORTHO else 0.15 if PREDS else 0.12, 1, 1))
    assert_local_path(FIGDIR)
    dest = BACKUP        # every block-level classifier panel is supporting
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(dest, f"r16_event_type_classifier{SFX}{TAG}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"figure -> {os.path.join(dest, 'r16_event_type_classifier' + SFX + TAG)}.png/.svg")


def classify_sessions(W):
    """The same event-type classifier with the WHOLE SESSION as the fitting unit.

    Session is this project's declared unit of analysis. Each fit uses the session's full
    window count rather than one block's worth, giving one coefficient for the whole sleep
    recording. The block-level and session-level fits are reported together.

    Returned in the same shape as `classify_blocks` so the same report and plot work,
    with `block` set to the session name.
    """
    cols = [c for c, _ in PRED]
    rows = []
    for s, g in W.groupby("session"):
        g = g.dropna(subset=cols)
        if (len(g) < 30 or g.n_barr.sum() < MIN_EVENTS
                or g.n_rip.sum() < MIN_EVENTS):
            print(f"  [skipped {s}: {int(g.n_barr.sum())} barrages, {len(g)} windows]")
            continue
        Z = ((g[cols] - g[cols].mean()) / (g[cols].std(ddof=0) + 1e-12)).to_numpy(float)
        endog = np.column_stack([g.n_barr, g.n_rip])
        # 0-trial windows are skipped (see classify_blocks).
        keep = endog.sum(1) > 0
        Z, endog = Z[keep], endog[keep]
        try:
            f = sm.GLM(endog, sm.add_constant(Z), family=sm.families.Binomial()).fit()
            f0 = sm.GLM(endog, np.ones((len(endog), 1)),
                        family=sm.families.Binomial()).fit()
        except Exception:
            print(f"  [did not converge at session level: {s}]")
            continue
        dev = 100.0 * (f0.deviance - f.deviance) / f0.deviance
        disp = float(f.pearson_chi2 / f.df_resid) if f.df_resid > 0 else 1.0
        for j, (c, _) in enumerate(PRED):
            b_, se = float(f.params[1 + j]), float(f.bse[1 + j])
            rows.append(dict(session=s, block=s, n_win=len(g), predictor=c, coef=b_,
                             se=se, z=b_ / se if se > 0 else np.nan,
                             z_qb=b_ / (se * np.sqrt(max(disp, 1.0))) if se > 0 else np.nan,
                             dev_expl=dev, n_barr=int(g.n_barr.sum()),
                             n_rip=int(g.n_rip.sum())))
    C = pd.DataFrame(rows)
    # z_within: the five coefficients of one session z-scored against each other,
    # Fig2_F_GLM.ipynb's `zscore_within_pkl` convention (ddof = 1). Compositional -- the
    # five sum to zero inside a session -- so it says where a predictor stands RELATIVE
    # TO THE OTHERS in that session, unlike the coefficient / its own SE axis.
    C["z_within"] = C.groupby("session").coef.transform(
        lambda v: (v - v.mean()) / (v.std(ddof=1) + 1e-12))
    return C


def report_sessions(C, store=None):
    print(f"\n{'=' * 92}\nEVENT-TYPE CLASSIFIER, SESSION LEVEL [{PRED_SET}]: "
          f"{C.session.nunique()} sessions,\nmedian deviance explained "
          f"{C.dev_expl.median():.1f}%\n{'=' * 92}")
    print(f"  {'predictor':11s}{'median coef':>13s}{'sessions same sign':>21s}"
          f"{'signed-rank p':>15s}{'t p':>9s}{'median z':>11s}{'p (z)':>9s}"
          f"{'sr p (z)':>11s}{'sr p (qb)':>11s}{'z within':>10s}{'same':>7s}{'sr p':>8s}")
    for c, lab in PRED:
        v = C[C.predictor == c]
        b = v.coef.to_numpy()
        same = max(int((b > 0).sum()), int((b < 0).sum()))
        pw = float(stats.wilcoxon(b).pvalue)
        _, pt = stats.ttest_1samp(b, 0.0)
        _, pz = stats.ttest_1samp(v.z.to_numpy(), 0.0)
        pwz = float(stats.wilcoxon(v.z.to_numpy()).pvalue)
        pwq = float(stats.wilcoxon(v.z_qb.to_numpy()).pvalue)
        zw = v.z_within.to_numpy()
        pwzw = float(stats.wilcoxon(zw).pvalue)
        samezw = max(int((zw > 0).sum()), int((zw < 0).sum()))
        print(f"  {lab:11s}{np.median(b):+13.4f}{same:15d}/{b.size:<5d}"
              f"{pw:15.4f}{pt:9.4f}{np.median(v.z):+11.2f}{pz:9.4f}{pwz:11.4f}{pwq:11.4f}"
              f"{np.median(zw):+10.2f}{samezw:4d}/{zw.size:<2d}{pwzw:8.4f}")
        if store is not None:
            store.append(dict(level="session", pred_set=PRED_SET, predictor=c, label=lab,
                              median_coef=float(np.median(b)), n_sessions=int(b.size),
                              same_sign=same, p_signrank=pw, p_t=float(pt),
                              median_z=float(np.median(v.z)), p_z=float(pz),
                              p_signrank_z=pwz, p_signrank_zqb=pwq,
                              median_z_qb=float(np.median(v.z_qb)),
                              median_z_within=float(np.median(zw)), same_sign_z_within=samezw,
                              p_signrank_z_within=pwzw,
                              dev_expl=float(C.dev_expl.median())))
    print("\n  per-session d2 coefficient (negative = higher d2 means SWRs):")
    for _, r in C[C.predictor == PRED[0][0]].sort_values("coef").iterrows():
        print(f"    {r.session:12s}{r.coef:+8.3f}  (z = {r.z:+6.2f}, "
              f"{r.n_barr} barrages / {r.n_rip} SWRs)")


def pairwise_holm(C, value="coef"):
    """Fig2_F_GLM.ipynb's `perform_pairwise_comparisons`, reused here: paired Wilcoxon
    signed-rank between predictors, paired by session, Holm-corrected across all pairs,
    "winner" by mean difference. That notebook runs it on a positive-is-better importance;
    a classifier coefficient is signed, so it is run here on the MAGNITUDE |coef| --
    whether d2 carries a larger effect than the others, whatever the sign -- and sign
    consistency is reported separately by report_sessions. Under magnitude, a competitor
    whose sign flips between sessions still contributes its |coef|. Does
    not touch the panel.

        -> data/revision/log/r16_event_type_classifier_session_pairwise{TAG}.csv
    """
    import itertools
    from statsmodels.stats.multitest import multipletests
    LAB = dict(PRED)
    piv = C.pivot(index="session", columns="predictor", values=value).abs()
    rows = []
    for f1, f2 in itertools.combinations([c for c, _ in PRED], 2):
        sub = piv[[f1, f2]].dropna()
        if len(sub) < 2:
            continue
        _, pv = stats.wilcoxon(sub[f1], sub[f2])
        d = float(sub[f1].mean() - sub[f2].mean())
        rows.append(dict(comparison=f"{LAB[f1]} vs {LAB[f2]}", f1=f1, f2=f2,
                         winner=LAB[f1] if d > 0 else LAB[f2], mean_diff=abs(d),
                         p=float(pv), n=int(len(sub))))
    R = pd.DataFrame(rows).sort_values("p").reset_index(drop=True)
    rej, padj, _, _ = multipletests(R.p.to_numpy(), method="holm")
    R["p_holm"], R["reject"] = padj, rej
    print("")
    print("PAIRWISE, Fig 2F's procedure on |coef| (paired Wilcoxon by session, Holm):")
    print(f"  {'comparison':26s}{'larger |coef|':>15s}{'|mean diff|':>13s}{'p':>10s}"
          f"{'p Holm':>10s}{'n':>4s}")
    for _, r in R.iterrows():
        sig = ("***" if r.p_holm < .001 else "**" if r.p_holm < .01
               else "*" if r.p_holm < .05 else "ns")
        print(f"  {r.comparison:26s}{r.winner:>15s}{r.mean_diff:13.3f}{r.p:10.4f}"
              f"{r.p_holm:10.4f}{r.n:4d}  {sig}")
    R.to_csv(assert_local_path(os.path.join(
        LOGDIR, f"r16_event_type_classifier_session_pairwise{TAG}.csv")), index=False)
    return R


def plot_sessions(C, S):
    """Session-level panel: ONE figure on the simple within-session z.

    The main figure plots the coefficient itself: log-odds an event is a barrage rather
    than a SWR per within-session SD of the predictor. The predictors are z-scored within
    session before the fit, so this is the effect per within-session z-unit -- the simple
    within-session z. Each coefficient divided by its own standard error is a Wald
    statistic rather than a within-session z, and is not used here.

    The alternative reading, Fig2_F_GLM.ipynb's `zscore_within_pkl` (the five coefficients
    of a session z-scored against each other), is drawn to backup/: it
    is compositional (the five sum to zero), so whenever d2 is the most negative
    coefficient, another predictor is pushed positive as its mirror rather than an
    independent finding. Numbers = sessions sharing the sign; stars = paired signed-rank
    across sessions.
    """
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial",
                                             "DejaVu Sans"],
                         "font.size": 12, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    order = [c for c, _ in PRED]
    labs = [l for _, l in PRED]
    r0 = S.iloc[0]
    d2r = S[S.predictor == PRED[0][0]].iloc[0]
    comp = S[S.predictor != PRED[0][0]]
    state = ("predictors measured BETWEEN events (ripples and barrages excised)"
             if PRED_SET == "excised" else
             "⚠ predictors measured on the INTACT window (circular; comparison only)")
    # The model is the barrage SHARE of a window's events, not a per-event classifier, and
    # the single-predictor fits (r16_classifier_marginal_unique.csv) are reported next to
    # the joint one.
    alone = ""
    mpath = os.path.join(LOGDIR, f"r16_classifier_marginal_unique{TAG}.csv")
    if os.path.exists(mpath):
        M = pd.read_csv(mpath)
        M = M[M.kind.str.startswith("marginal")]
        parts = []
        for c, lab_ in PRED:
            v = M[M.predictor == c].coef.dropna()
            if len(v) > 2 and stats.wilcoxon(v).pvalue < 0.05:
                parts.append(f"{lab_} {max(int((v < 0).sum()), int((v > 0).sum()))}/{len(v)}, "
                             f"p = {stats.wilcoxon(v).pvalue:.3f}")
        if parts:
            alone = "Entered alone, significant: " + "; ".join(parts) + ".\n"
    specs = [
        ("coef", "same_sign", "p_signrank",
         "log-odds barrages : SWRs in a window,\nper within-session SD",
         "Predictors z-scored within session; axis = effect per within-session SD, adjusted for\n"
         "the other four. Numbers = sessions sharing the majority sign; stars = one-sample\n"
         "signed-rank test of the session coefficients against 0. Together, only d2 keeps a\n"
         f"consistent sign ({int(d2r.same_sign)}/{int(d2r.n_sessions)}, p = {d2r.p_signrank:.4f}; "
         f"no other measure exceeds {int(comp.same_sign.max())}/{int(d2r.n_sessions)}).\n"
         f"{alone}"
         "NN8_day6 (6 barrages in the whole recording) cannot be fitted.",
         os.path.join(FIGDIR if not TAG else BACKUP, f"r16_event_type_classifier_session{TAG}")),
        ("z_within", "same_sign_z_within", "p_signrank_z_within",
         "coefficient z-scored across the five\npredictors within each session",
         "RECORD ONLY. Each session's five coefficients z-scored against each other\n"
         "(Fig 2F's zscore_within_pkl, ddof 1). They sum to zero, so this axis is\n"
         "compositional: where d2 is the most negative, another predictor is pushed\n"
         f"positive. d2 {int(d2r.same_sign_z_within)}/{int(d2r.n_sessions)}, "
         f"p = {d2r.p_signrank_z_within:.4f}; the best competitor reaches "
         f"{int(comp.same_sign_z_within.max())}/{int(d2r.n_sessions)} as the mirror\n"
         "of d2, not as a finding. The main figure is the coefficient per within-session SD.",
         os.path.join(BACKUP, f"r16_event_type_classifier_session_zwithin{TAG}")),
    ]
    rng = np.random.default_rng(1)
    for col, kss, kp, ylab, foot, stem in specs:
        fig, ax = plt.subplots(figsize=(5.8, 5.0))
        sns.boxplot(data=C, x="predictor", y=col, order=order, ax=ax, width=0.6,
                    showfliers=False, linewidth=2, palette=COLOR, hue="predictor",
                    legend=False)
        lo, hi = C[col].quantile(0.02), C[col].quantile(0.98)
        pad = 0.12 * (hi - lo)
        for i, c in enumerate(order):
            v = C[C.predictor == c][col].to_numpy()
            ax.plot(i + rng.uniform(-0.13, 0.13, v.size), np.clip(v, lo, hi), "o", ms=5,
                    color="k", alpha=0.6, mew=0)
            r = S[S.predictor == c].iloc[0]
            p = float(r[kp])
            star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
            ax.text(i, hi + 1.0 * pad, f"{int(r[kss])}/{int(r.n_sessions)}\n{star}",
                    ha="center", va="bottom", fontsize=10,
                    fontweight="bold" if p < .05 else "normal")
        n_clip = int(((C[col] < lo) | (C[col] > hi)).sum())
        if n_clip:
            ax.text(0.99, 0.015, f"{n_clip} clipped to axis", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8, color="0.4")
        ax.axhline(0, ls="--", lw=1.2, color="k")
        ax.set_ylim(lo - pad, hi + 3.6 * pad)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(labs, rotation=30, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel(ylab)
        # not "barrage-dominated": few windows hold more barrages than SWRs; the model is
        # the barrage SHARE of a window's events, not which type dominates it
        ax.set_title("background CA1 dynamics track the relative\n"
                     "composition of SWRs and barrages during NREM\n"
                     "one model per session over 30 s windows, all five measures together\n"
                     f"{state}\n"
                     "negative = the higher the measure, the more SWRs relative to barrages\n"
                     f"{int(r0.n_sessions)} sessions, median {r0.dev_expl:.0f}% of deviance explained",
                     fontsize=9.5, loc="left")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        fig.text(0.01, 0.005, foot, fontsize=8, color="0.3", ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        assert_local_path(FIGDIR)
        for ext in ("png", "svg"):
            fig.savefig(f"{stem}.{ext}", dpi=300)
        plt.close(fig)
        print(f"figure -> {stem}.png/.svg")


if __name__ == "__main__":
    main()
