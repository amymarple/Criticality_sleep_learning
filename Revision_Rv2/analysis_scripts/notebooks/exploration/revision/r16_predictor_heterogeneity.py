"""
r16_predictor_heterogeneity.py -- across-block heterogeneity of each predictor's event-type
coefficient (MUA, d2, ...) in the ORTHO block classifier.

The block classifier (r16_block_level_dissociation.py, ORTHO=1: one binomial GLM per 30 min
block of NREM, barrages against SWRs, predictors on event-free time, competitors residualised
on one another so their coefficients are not variance-inflated) gives MUA and d2 both negative
session-mean coefficients, but a much larger between-session spread for MUA. Three diagnostics,
all from stored tables (no raw-data pass):

  1. NOISE OR HETEROGENEITY. Per block the fit gives a coefficient and its SE, so the scatter
     of the coefficients can be compared with what their SEs allow: Cochran's Q, I^2 and the
     between-block SD tau of a random-effects meta-analysis, with the fixed- and
     random-effects pooled estimate, for each predictor.
  2. THE PREDICTOR'S OWN WITHIN-BLOCK STRUCTURE. Coefficient of variation inside a block,
     share of within-block variance that is a slow trend (R^2 on a cubic in time), and
     lag-1 autocorrelation, for every predictor.
  3. CELL-TYPE COMPOSITION. MUA_bg = n_e r_E + n_i r_I and EI_bg = log(r_E / r_I) are stored
     per window (every unit is E or I, so this identity is exact and is asserted), which
     gives r_E and r_I per window. The classifier is refit with the E rate and the I rate in
     place of MUA + E/I (the same information, split by cell type), plus each rate alone, to
     test whether MUA's sign tracks the E/I composition of the recorded units.

Unit of inference throughout: t over session means of block coefficients, as in
`r16_block_level_dissociation.report_classify`; blocks of >= 60 windows with >= MIN_EVENTS of
both event types.

Inputs: r16_event_type_classifier_ortho.csv, r16_event_excised_windows.csv (via `load`).

    -> data/revision/log/r16_predictor_heterogeneity.csv           (diagnostic 1 + 2)
    -> data/revision/log/r16_predictor_heterogeneity_ei_split.csv  (diagnostic 3)
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
from r16_block_level_dissociation import load, blocks, LOGDIR, MIN_EVENTS  # noqa: E402

MINW = 60
ORTHO_CSV = os.path.join(LOGDIR, "r16_event_type_classifier_ortho.csv")
LAB_ORTHO = {"d2_bg": "d2", "MUA_bg": "MUA", "EI_bg": "E/I | MUA", "fano_bg": "Fano | MUA, E/I",
             "sync_bg": "synchrony | MUA, E/I, Fano"}
LAB = {"d2_bg": "d2", "MUA_bg": "MUA", "EI_bg": "E/I", "fano_bg": "MUA Fano", "sync_bg": "synchrony",
       "E_tot": "E rate", "I_tot": "I rate"}


def meta(b, se):
    """Fixed- and random-effects (DerSimonian-Laird) pooling of per-block coefficients."""
    w = 1 / se ** 2
    fe = np.sum(w * b) / np.sum(w)
    q = float(np.sum(w * (b - fe) ** 2))
    df = b.size - 1
    c = np.sum(w) - np.sum(w ** 2) / np.sum(w)
    tau2 = max(0.0, (q - df) / c)
    wr = 1 / (se ** 2 + tau2)
    re = np.sum(wr * b) / np.sum(wr)
    return dict(Q=q, df=df, p_het=float(1 - stats.chi2.cdf(q, df)),
                I2=100 * max(0.0, (q - df) / q), tau=float(np.sqrt(tau2)),
                fe_coef=float(fe), fe_z=float(fe * np.sqrt(np.sum(w))),
                re_coef=float(re), re_z=float(re * np.sqrt(np.sum(wr))))


def good_blocks(W, cols, minw=MINW):
    for s, g in W.groupby("session"):
        g = blocks(g.dropna(subset=cols), minw)
        for bid, h in g.groupby("blk"):
            if len(h) < minw or h.n_barr.sum() < MIN_EVENTS or h.n_rip.sum() < MIN_EVENTS:
                continue
            yield s, f"{s}#{bid}", h


def within_block(W, cols):
    rows = []
    for s, blk, h in good_blocks(W, cols):
        t = (h.w - h.w.mean()).to_numpy(float)
        t /= t.std()
        X = np.column_stack([np.ones(len(h)), t, t ** 2, t ** 3])
        for c in cols:
            y = h[c].to_numpy(float)
            bb = np.linalg.lstsq(X, y, rcond=None)[0]
            rows.append(dict(session=s, block=blk, predictor=c,
                             cv=y.std() / abs(y.mean()) if y.mean() != 0 else np.nan,
                             trend_r2=1 - np.sum((y - X @ bb) ** 2) / np.sum((y - y.mean()) ** 2),
                             ac1=np.corrcoef(y[:-1], y[1:])[0, 1]))
    return pd.DataFrame(rows)


def classify(W, cols):
    rows = []
    for s, blk, h in good_blocks(W, cols):
        Z = ((h[cols] - h[cols].mean()) / (h[cols].std(ddof=0) + 1e-12)).to_numpy(float)
        try:
            endog = np.column_stack([h.n_barr, h.n_rip])
            keep = endog.sum(1) > 0     # 0-trial windows skipped (see r16_block_level_dissociation)
            f = sm.GLM(endog[keep], sm.add_constant(Z[keep]),
                       family=sm.families.Binomial()).fit()
        except Exception:
            continue
        for j, c in enumerate(cols):
            rows.append(dict(session=s, block=blk, predictor=c, coef=float(f.params[1 + j]),
                             se=float(f.bse[1 + j])))
    return pd.DataFrame(rows)


def session_test(v):
    gm = v.groupby("session").coef.mean().to_numpy()
    t, p = stats.ttest_1samp(gm, 0.0)
    b = v.coef.to_numpy()
    return dict(median_coef=float(np.median(b)), n_blocks=int(b.size),
                blocks_same=max(int((b > 0).sum()), int((b < 0).sum())),
                n_sess=int(gm.size), sess_same=max(int((gm > 0).sum()), int((gm < 0).sum())),
                sess_mean=float(gm.mean()), sess_sd=float(gm.std(ddof=1)), t=float(t), p=float(p))


def main():
    W = load()
    C = pd.read_csv(ORTHO_CSV)
    preds = ["d2_bg", "MUA_bg", "EI_bg", "fano_bg", "sync_bg"]

    print(f"1. Heterogeneity of the block coefficients (ORTHO classifier, {C.block.nunique()} blocks):")
    print(f"   {'predictor':28s}{'med|z|':>7s}{'p<.05':>7s}{'Q':>8s}{'df':>4s}{'I2':>5s}{'tau':>7s}"
          f"{'FE coef':>9s}{'FE z':>7s}{'RE coef':>9s}{'RE z':>7s}")
    out = []
    for c in preds:
        v = C[C.predictor == c]
        b, se = v.coef.to_numpy(), v.se.to_numpy()
        m = meta(b, se)
        m.update(predictor=c, label=LAB_ORTHO[c], median_abs_z=float(np.median(np.abs(b / se))),
                 blocks_p05=int((np.abs(b / se) > 1.96).sum()), n_blocks=int(b.size))
        out.append(m)
        print(f"   {LAB_ORTHO[c]:28s}{m['median_abs_z']:7.2f}{m['blocks_p05']:4d}/{b.size:<3d}{m['Q']:8.1f}"
              f"{m['df']:4d}{m['I2']:5.0f}{m['tau']:7.3f}{m['fe_coef']:+9.3f}{m['fe_z']:+7.2f}"
              f"{m['re_coef']:+9.3f}{m['re_z']:+7.2f}")
    H = pd.DataFrame(out)

    print("\n2. The predictor inside a 30 min block (medians over blocks):")
    raw = ["d2_bg", "MUA_bg", "fano_bg", "sync_bg", "EI_bg"]
    S = within_block(W, raw).groupby("predictor")[["cv", "trend_r2", "ac1"]].median()
    S.columns = ["within_block_cv", "within_block_trend_r2", "within_block_ac1"]
    print(S.round(3).to_string())
    H = H.merge(S, left_on="predictor", right_index=True, how="left")
    assert_local_path(LOGDIR)
    H.to_csv(assert_local_path(os.path.join(LOGDIR, "r16_predictor_heterogeneity.csv")), index=False)

    print("\n3. Cell-type composition: MUA split into the E rate and the I rate")
    ex = np.exp(W.EI_bg)
    W["rI"] = W.MUA_bg / (W.n_e * ex + W.n_i)
    W["rE"] = W.rI * ex
    W["E_tot"], W["I_tot"] = W.rE * W.n_e, W.rI * W.n_i
    err = float((W.E_tot + W.I_tot - W.MUA_bg).abs().max())
    assert err < 1e-8, f"E + I does not reproduce MUA_bg (max |diff| {err:.2e})"
    print(f"   identity E_tot + I_tot = MUA_bg holds to {err:.1e} Hz (every unit is E or I)")
    rEI = W.groupby("session").apply(lambda h: h.E_tot.corr(h.I_tot), include_groups=False)
    print(f"   within-session r(E rate, I rate): median {rEI.median():+.2f}, range {rEI.min():+.2f} .. {rEI.max():+.2f}")
    rows = []
    for cols, name in [(["d2_bg", "E_tot", "I_tot", "fano_bg", "sync_bg"], "joint (E and I rates in place of MUA + E/I)"),
                       (["d2_bg", "MUA_bg", "fano_bg", "sync_bg", "EI_bg"], "joint (raw five, for reference)"),
                       (["E_tot"], "E rate alone"), (["I_tot"], "I rate alone"), (["MUA_bg"], "MUA alone")]:
        R = classify(W, cols)
        print(f"   {name}  [{R.block.nunique()} blocks, {R.session.nunique()} sessions]")
        for c in cols:
            r = session_test(R[R.predictor == c])
            rows.append(dict(model=name, predictor=c, label=LAB.get(c, c), **r))
            print(f"     {LAB.get(c, c):10s} median {r['median_coef']:+.3f}  blocks {r['blocks_same']}/{r['n_blocks']}"
                  f"  sessions {r['sess_same']}/{r['n_sess']}  session-mean {r['sess_mean']:+.3f} (SD {r['sess_sd']:.3f})"
                  f"  t {r['t']:+.2f}  p {r['p']:.4f}")
    pd.DataFrame(rows).to_csv(assert_local_path(
        os.path.join(LOGDIR, "r16_predictor_heterogeneity_ei_split.csv")), index=False)

    intf = W.groupby("session").apply(lambda h: (h.n_i / (h.n_e + h.n_i)).mean(), include_groups=False)
    gm = C[C.predictor == "MUA_bg"].groupby("session").coef.mean()
    rho, p = stats.spearmanr(intf.loc[gm.index], gm)
    keep = gm.index != "NN16_day13"
    rho2, p2 = stats.spearmanr(intf.loc[gm.index][keep], gm[keep])
    print(f"\n   session-mean MUA coefficient vs interneuron fraction (n = {len(gm)}): Spearman rho {rho:+.2f}, "
          f"p = {p:.3f}; without NN16_day13 rho {rho2:+.2f}, p = {p2:.3f}  (descriptive)")
    print(f"\n-> {LOGDIR}\\r16_predictor_heterogeneity.csv, _ei_split.csv")


if __name__ == "__main__":
    main()
