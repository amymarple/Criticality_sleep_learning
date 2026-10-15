"""
ed2a_novel_vs_control.py -- ED2a as a difference-in-differences contrast between novel
and familiar-control sessions, rather than a within-arm post-mid change.

FigS2_A_novel_familar tests each arm separately: LMM d2_rel ~ S2 vs S1 within the novel
days, and within the control days. On a novel day S1 (midsleep) follows a familiar task
and S2 (postsleep) the novel task; on a control day both phases follow familiar tasks.
The contrast used instead is
  DiD = (S2 - S1)_novel - (S2 - S1)_control.

Design: per-session values as the notebook's LMM (median d2_rel per file x session; mean
as a check), d2 (2024 Euclidean, p = 20, 40 ms), d2_rel = (d2 - m_pre) / (d2 + m_pre)
relative to pre-task sleep. Tables: all windows (`ed2a_corrected_all_windows.csv`,
PPP8_day14 from its own d2, the primary session-level test), shipped (reference), disjoint
(`ed_nonoverlap_tables`). Files with both S1 and S2 only (PPP7_day14 has no S1). Tests, all
with the session as the unit:
  1. LMM value ~ C(S2) * C(novel), random intercept per file -> the interaction = DiD;
  2. the same with animal as the group and file as a variance component (5 animals have
     both kinds);
  3. exact permutation of the novel / control labels over the per-session Delta (all
     C(n, k) assignments), plus Mann-Whitney;
  4. within animal: mean Delta(novel) - mean Delta(control) for each animal that has both.

    -> data/revision/log/ed2a_novel_vs_control.csv
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy.stats import mannwhitneyu, wilcoxon  # noqa: E402

warnings.filterwarnings("ignore")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
TABLES = {
    "all windows, PPP8_day14 recomputed (primary)": os.path.join(LOGDIR, "ed2a_corrected_all_windows.csv"),
    "shipped (PPP8_day14 = day-12 data)":
        r"D:\Documents\GitHub\Criticality_sleep_learning\Previous_Rv1\csv\FigS2_A_novel_familar\d2_rel_novel_familiar.csv",
    "disjoint, PPP8_day14 recomputed": os.path.join(LOGDIR, "ed_nonoverlap_tables", "d2_rel_novel_familiar.csv"),
}


def per_session(T, agg):
    s = T.groupby(["file", "source", "phase"])["d2_rel"].agg(agg).unstack("phase").dropna(subset=["mid", "post"])
    s = s.reset_index()
    s["delta"] = s["post"] - s["mid"]
    s["animal"] = s["file"].str.split("_").str[0]
    s["novel"] = (s["source"] == "novel").astype(int)
    return s


def exact_perm(delta, is_novel):
    d = np.asarray(delta, float)
    k = int(is_novel.sum())
    obs = d[is_novel == 1].mean() - d[is_novel == 0].mean()
    tot = d.sum()
    n = len(d)
    diffs = []
    for idx in itertools.combinations(range(n), k):
        sn = d[list(idx)].sum()
        diffs.append(sn / k - (tot - sn) / (n - k))
    diffs = np.asarray(diffs)
    return obs, float(np.mean(np.abs(diffs) >= abs(obs) - 1e-12)), len(diffs)


def lmm_interaction(s):
    L = s.melt(id_vars=["file", "animal", "novel"], value_vars=["mid", "post"], var_name="phase", value_name="v")
    L["S2"] = (L.phase == "post").astype(int)
    out = {}
    try:
        r = smf.mixedlm("v ~ S2 * novel", L, groups=L["file"]).fit(method="lbfgs", reml=True)
        out["lmm_file_did"], out["lmm_file_se"], out["lmm_file_p"] = r.params["S2:novel"], r.bse["S2:novel"], r.pvalues["S2:novel"]
    except Exception as e:                                    # noqa: BLE001
        out["lmm_file_err"] = str(e)
    try:
        r = smf.mixedlm("v ~ S2 * novel", L, groups=L["animal"], vc_formula={"file": "0 + C(file)"}).fit(method="lbfgs", reml=True)
        out["lmm_animal_did"], out["lmm_animal_se"], out["lmm_animal_p"] = r.params["S2:novel"], r.bse["S2:novel"], r.pvalues["S2:novel"]
    except Exception as e:                                    # noqa: BLE001
        out["lmm_animal_err"] = str(e)
    return out


def main():
    rows = []
    for label, path in TABLES.items():
        T = pd.read_csv(path, float_precision="round_trip")
        for agg in ("median", "mean"):
            s = per_session(T, agg)
            nov, con = s[s.novel == 1], s[s.novel == 0]
            obs, p_perm, n_perm = exact_perm(s.delta.to_numpy(), s.novel.to_numpy())
            mw = mannwhitneyu(nov.delta, con.delta, alternative="two-sided")
            a = s.groupby(["animal", "novel"]).delta.mean().unstack("novel").dropna()
            within = (a[1] - a[0])
            r = dict(table=label, per_session=agg, n_novel=len(nov), n_control=len(con),
                     delta_novel_mean=nov.delta.mean(), delta_control_mean=con.delta.mean(),
                     delta_novel_median=nov.delta.median(), delta_control_median=con.delta.median(),
                     did_mean=obs, p_perm_exact=p_perm, n_perm=n_perm, p_mannwhitney=mw.pvalue,
                     n_novel_up=int((nov.delta > 0).sum()), n_control_up=int((con.delta > 0).sum()),
                     n_animals_both=len(within), within_animal_pos=int((within > 0).sum()),
                     within_animal_mean=float(within.mean()) if len(within) else np.nan,
                     p_within_wilcoxon=float(wilcoxon(within).pvalue) if len(within) >= 3 else np.nan,
                     **lmm_interaction(s))
            rows.append(r)
            print(f"\n{label} | per-session {agg}: novel n={len(nov)} (up {r['n_novel_up']}), control n={len(con)} "
                  f"(up {r['n_control_up']})")
            print(f"  Delta(post-mid) mean: novel {r['delta_novel_mean']:+.4f}, control {r['delta_control_mean']:+.4f}  "
                  f"-> DiD {obs:+.4f}; exact permutation p = {p_perm:.4f} ({n_perm} assignments); Mann-Whitney p = {mw.pvalue:.4f}")
            print(f"  LMM interaction (file RE): {r.get('lmm_file_did', np.nan):+.4f} +/- {r.get('lmm_file_se', np.nan):.4f}, "
                  f"p = {r.get('lmm_file_p', np.nan):.4f} | (animal RE, file VC): {r.get('lmm_animal_did', np.nan):+.4f}, "
                  f"p = {r.get('lmm_animal_p', np.nan):.4f}")
            print(f"  within animal ({len(within)} animals with both): novel - control Delta > 0 in "
                  f"{r['within_animal_pos']}/{len(within)}, mean {r['within_animal_mean']:+.4f}, Wilcoxon p = {r['p_within_wilcoxon']:.3f}; "
                  + ", ".join(f"{k} {v:+.3f}" for k, v in within.items()))
    pd.DataFrame(rows).to_csv(assert_local_path(os.path.join(LOGDIR, "ed2a_novel_vs_control.csv")), index=False)


if __name__ == "__main__":
    main()
