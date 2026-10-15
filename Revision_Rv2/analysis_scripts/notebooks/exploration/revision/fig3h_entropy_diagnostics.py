"""fig3h_entropy_diagnostics.py -- Figure 3h (ISI entropy vs barrage rate): the sextile-mean trend
compared with the per-session level, and the mixed model drawn as the panel's star.

The per-session Spearman test of fig3_background_repro.py (non-overlapping windows) and the
sextile-mean trend can differ; this separates the candidate causes on the stored windows (`data/ripple_barr/ISI_entropy/barrage/*.pkl`: isi_entropy, n_spikes,
ripple_rate, barrage_rate per 30 s NREM window), non-overlapping subset (>= 30 s apart):
  * zeros: rho on windows with >= 1 barrage only; entropy with vs without barrages;
  * firing rate: rho(entropy, n_spikes), rho(n_spikes, barrage rate), rho partial on n_spikes;
  * SWR rate: partial on ripple_rate;
  * barrage-poor sessions: rho against each session's barrage count;
  * the published tiling: which sessions populate each `add_session_tiles` sextile (sliding windows).

  * linear mixed models, windows as rows, session as the group: value ~ barrage rate + (1 | session);
    + random slope (1 + rate | session); + spike count. Panel e (background d2,
    `r16_glue_control_windows.csv`) is fitted the same way for comparison.

    -> data/revision/log/fig3h_entropy_diagnostics.csv        (per session)
    -> data/revision/log/fig3h_entropy_diagnostics_lmm.csv    (LMM coefficients)
    -> data/revision/log/fig3h_entropy_diagnostics_tiles.csv  (published tiling: session x tile)
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                          # noqa: E402
import fig3_background_repro as F                                       # noqa: E402  (notebook functions)


def pspear(x, y, z):
    """Spearman of x and y, partial on z (ranks, linear residuals)."""
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    A = np.c_[np.ones_like(rz), rz]
    ex = rx - A @ np.linalg.lstsq(A, rx, rcond=None)[0]
    ey = ry - A @ np.linalg.lstsq(A, ry, rcond=None)[0]
    return float(np.corrcoef(ex, ey)[0, 1])


def main():
    d = F.load_pooled_per_center(os.path.join(F.RB, "ISI_entropy", "barrage"), state_filter="NREM")
    d["sess"] = d.animal_name.astype(str) + "_" + d.day.astype(str)
    N = pd.concat([F.nonoverlap(g.sort_values("center_s")) for _, g in d.groupby("sess")]).reset_index(drop=True)
    rows = []
    for s, g in N.groupby("sess"):
        br, nz = g.barrage_rate, g.barrage_rate > 0
        rows.append(dict(
            session=s, n_windows=len(g), frac_zero=float((~nz).mean()), n_barrages=int(round((br * 30).sum())),
            rho=stats.spearmanr(g.isi_entropy, br)[0],
            rho_nonzero=(stats.spearmanr(g.isi_entropy[nz], br[nz])[0] if nz.sum() >= 10 and br[nz].nunique() > 2 else np.nan),
            ent_with_minus_without=(g.isi_entropy[nz].mean() - g.isi_entropy[~nz].mean()) if nz.any() else np.nan,
            rho_entropy_nspikes=stats.spearmanr(g.isi_entropy, g.n_spikes)[0],
            rho_nspikes_barrage=stats.spearmanr(g.n_spikes, br)[0],
            rho_partial_nspikes=pspear(g.isi_entropy.values, br.values, g.n_spikes.values),
            rho_partial_ripple=pspear(g.isi_entropy.values, br.values, g.ripple_rate.values)))
    T = pd.DataFrame(rows).sort_values("n_barrages")
    T.to_csv(assert_local_path(os.path.join(F.LOG, "fig3h_entropy_diagnostics.csv")), index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string(index=False))
    print("\nper session (non-overlapping windows): median, n positive, signed-rank p")
    for c in ("rho", "rho_nonzero", "ent_with_minus_without", "rho_partial_nspikes", "rho_partial_ripple",
              "rho_entropy_nspikes", "rho_nspikes_barrage"):
        v = T[c].dropna()
        print(f"  {c:24s} {v.median():+.3f}  {(v > 0).sum()}/{len(v)}  p = {stats.wilcoxon(v).pvalue:.3f}")
    print(f"  Spearman(rho, n_barrages) across sessions = {stats.spearmanr(T.rho, T.n_barrages)[0]:+.2f}")

    t = F.add_session_zscores(F.add_session_tiles(d.copy(), by="barrage", k=6), "isi_entropy", robust=False)
    P = t.groupby(["sess", "tile_barrage_sess"]).isi_entropy_z.mean().unstack()
    P.to_csv(assert_local_path(os.path.join(F.LOG, "fig3h_entropy_diagnostics_tiles.csv")))
    print("\npublished tiling, sessions per tile:", P.notna().sum().astype(int).to_dict())
    lmm_table(N)
    print("-> fig3h_entropy_diagnostics.csv, fig3h_entropy_diagnostics_tiles.csv, fig3h_entropy_diagnostics_lmm.csv")


def fit(df, y, x, extra="", slope=False, animal=True):
    """y ~ x (+ extra) (+ C(animal)), random intercept (and slope on x) per session. Animal enters as a
    fixed effect with session as the random-effect group, as in the Fig. 3 notebook's own LMMs
    (`d2 ~ C(epoch) + C(animal)`, groups = animal_day)."""
    import statsmodels.formula.api as smf
    D = df.dropna(subset=[y, x] + ([extra] if extra else [])).copy()
    D["x10"] = D[x] * 10                                  # coefficient per 0.1 Hz of barrage rate
    D["animal"] = D.session.str.split("_").str[0]
    f = f"{y} ~ x10" + (f" + {extra}" if extra else "") + (" + C(animal)" if animal else "")
    m = smf.mixedlm(f, D, groups=D["session"], re_formula="~x10" if slope else "1")
    r = m.fit(method="lbfgs", reml=True)
    lo, hi = r.conf_int().loc["x10"].to_list()
    return dict(coef_per_0p1Hz=float(r.params["x10"]), ci_lo=lo, ci_hi=hi, z=float(r.tvalues["x10"]),
                p=float(r.pvalues["x10"]), n_windows=int(r.nobs), n_sessions=int(D.session.nunique()),
                n_animals=int(D.animal.nunique()), converged=bool(r.converged))


def lmm_table(N):
    H = N.rename(columns={"sess": "session"})
    H["n_spikes_z"] = H.groupby("session").n_spikes.transform(lambda v: (v - v.mean()) / v.std())
    E = pd.read_csv(os.path.join(F.LOG, "r16_glue_control_windows.csv"))
    rows = []
    for panel, df, y in (("h ISI entropy (bits)", H, "isi_entropy"), ("e background d2", E, "d2_segments")):
        for lab, kw in (("(1|session)", dict(animal=False)),
                        ("C(animal) + (1|session)", {}),
                        ("C(animal) + (1 + rate|session)", {"slope": True}),
                        ("C(animal) + spike count + (1|session)", {"extra": "n_spikes_z"})):
            if "spike" in lab and panel.startswith("e"):
                continue
            rows.append(dict(panel=panel, model=lab, **fit(df, y, "barrage_rate", **kw)))
    L = pd.DataFrame(rows)
    L.to_csv(assert_local_path(os.path.join(F.LOG, "fig3h_entropy_diagnostics_lmm.csv")), index=False)
    print("\nlinear mixed models, non-overlapping NREM windows; coefficient per 0.1 Hz barrage rate")
    for _, r in L.iterrows():
        print(f"  {r.panel:22s} {r.model:40s} {r.coef_per_0p1Hz:+.5f} [{r.ci_lo:+.5f}, {r.ci_hi:+.5f}]  "
              f"z = {r.z:+.2f}  p = {r.p:.3g}  ({r.n_windows} windows, {r.n_sessions} sessions, {r.n_animals} animals"
              f"{'' if r.converged else ', NOT converged'})")


if __name__ == "__main__":
    main()
