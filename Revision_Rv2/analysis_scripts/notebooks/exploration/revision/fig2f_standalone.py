"""fig2f_standalone.py -- Figure 2f: replay rate, ablation of each background predictor.

Same content as the rate-model panel of r35_replay_lmm.plot, drawn as a standalone figure:

  bars      deviance explained lost when the predictor is dropped from the joint Poisson
            GLM of n_replay per 30 s window (session fixed effects, predictors z-scored
            within session), full - ablated, in percentage points; +- 95% cluster
            bootstrap CI over sessions, from r35_replay_rate_model.csv
            (r35_replay_rate_model.py, B = 1000)
  dots      the same drop-one per session (a session above the axis is drawn as an open
            triangle at the top with its value, never clipped silently)
  stars     the linear mixed model's fixed-effect z for that predictor (r35_replay_lmm:
            log(n_replay + 0.5), animal fixed, session random intercept + a random slope
            per session for every predictor)
  brackets  the mixed model's Wald contrast b_d2 - b_k, Holm over the three

Predictors: d2 (2024 Euclidean, p = 20, ripple-excised), MUA (bg), FR std | MUA (bg) and
E/I (bg) on the ripple-excised background of decoded NREM; nine sessions, six mice.

`compute()` and `draw(ax, D, compact=...)` are importable: fig2_composite.py draws the
same panel into the Figure 2 composite.

    -> reports/revision_figs/fig2f_standalone_excised.png / .svg
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from scipy import stats                                               # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                        # noqa: E402
from r35_replay_ablation_lrt import holm, COLOR, LOGDIR, FIGDIR       # noqa: E402
from r35_replay_lmm import prepare, fit_lmm, star, X, LAB, PRED       # noqa: E402


def compute():
    """Pooled drop-one + bootstrap CI + per-session values (rate_model CSV) and the mixed
    model's fixed effects and d2-vs-each contrasts (refit here)."""
    src = os.path.join(LOGDIR, "r35_replay_rate_model.csv")
    assert os.path.exists(src), "run r35_replay_rate_model.py first"
    G = pd.read_csv(src)
    G = G[G.model == "rate"]
    pooled = G[G.session.isna() & G.contrast.isna()].set_index("predictor")
    per = G[G.session.notna()].pivot_table(index="session", columns="predictor", values="drop_one_session")
    raw = [c for c, _ in PRED]
    assert all(c in pooled.index for c in raw) and all(c in per.columns for c in raw), "rate_model CSV predates the E/I set"
    per = per[raw]
    W = prepare()
    r = fit_lmm(W, "y_rate", True)
    C = r.cov_params().loc[X, X]
    con = []
    for k in X[1:]:
        e = float(r.params[X[0]] - r.params[k])
        se = float(np.sqrt(C.loc[X[0], X[0]] + C.loc[k, k] - 2 * C.loc[X[0], k]))
        z = e / se
        con.append((k, e, z, float(2 * stats.norm.sf(abs(z)))))
    ph = holm([p for *_, p in con])
    return dict(pooled=pooled, per=per, raw=raw, r=r, con=con, ph=ph,
                n_sess=W.session.nunique(), n_animal=W.animal.nunique(), n_win=len(W))


def report(D):
    pooled, raw, r = D["pooled"], D["raw"], D["r"]
    print(f"{D['n_win']} windows, {D['n_sess']} sessions, {D['n_animal']} mice; "
          f"joint GLM explains {pooled.loc[raw[0], 'dev_full']:.2f}%")
    for c, k in zip(raw, X):
        print(f"  {LAB[k]:20s} drop-one {pooled.loc[c, 'drop_one']:.2f} pp "
              f"[{pooled.loc[c, 'bs_lo']:.2f}, {pooled.loc[c, 'bs_hi']:.2f}]; LMM z = {r.tvalues[k]:+.1f} {star(r.pvalues[k])}")
    for (k, e, z, p), padj in zip(D["con"], D["ph"]):
        print(f"  d2 - {LAB[k]:18s} d(beta) = {e:+.3f}, z = {z:+.2f}, Holm p = {padj:.4f} {star(padj)}")


def draw(ax, D, compact=False):
    """The panel on a given axes. compact=True: the composite's smaller fonts and no
    footnote (the composite's caption carries the conventions)."""
    pooled, per, raw, r, con, ph = (D[k] for k in ("pooled", "per", "raw", "r", "con", "ph"))
    K = len(raw)
    cols = [COLOR[c] for c in raw]
    rng = np.random.default_rng(0)
    fs_lab, fs_br, fs_tick = (8.5, 7.5, 9) if compact else (10, 9, None)
    v = pooled.loc[raw, "drop_one"].to_numpy(float)
    lo, hi = pooled.loc[raw, "bs_lo"].to_numpy(float), pooled.loc[raw, "bs_hi"].to_numpy(float)
    ax.bar(range(K), v, width=0.6, color=cols, edgecolor="k", linewidth=1.5 if compact else 1.8, zorder=2)
    ax.errorbar(range(K), v, yerr=[v - lo, hi - v], fmt="none", ecolor="k", elinewidth=1.5 if compact else 1.8,
                capsize=4 if compact else 5, capthick=1.5 if compact else 1.8, zorder=4)
    top = max(hi.max(), np.percentile(per.to_numpy(float), 98))
    for j, (c, k) in enumerate(zip(raw, X)):
        pv = per[c].to_numpy(float)
        xs = j + rng.uniform(-0.13, 0.13, pv.size)
        inside = pv <= top
        ax.plot(xs[inside], pv[inside], "o", ms=3.5 if compact else 4.5, color="0.25", alpha=0.75, mew=0, zorder=3)
        for x_, y_, s_ in zip(xs[~inside], pv[~inside], per.index[~inside]):
            ax.plot(x_, top, "^", ms=5 if compact else 6, mfc="white", mec="0.25", mew=1.2, zorder=3)
            ax.annotate(f"{s_}: {y_:.1f}", (x_, top), xytext=(5, -2), textcoords="offset points",
                        fontsize=6 if compact else 7, color="0.35", ha="left", va="top")
        ax.text(j, top * 1.05, f"{v[j]:.2f}\n{star(r.pvalues[k])}", ha="center", va="bottom", fontsize=fs_lab)
    for q, ((k, e, z, p), padj) in enumerate(zip(con, ph)):
        j = X.index(k)
        y = top * (1.42 + 0.19 * q)
        ax.plot([0, 0, j, j], [y - 0.03 * top, y, y, y - 0.03 * top], color="k", lw=1.1, clip_on=False)
        ptxt = "p < 0.001" if padj < 0.001 else f"p = {padj:.3f}"
        ax.text(j / 2, y + 0.015 * top, f"{ptxt} {star(padj)}", ha="center", va="bottom", fontsize=fs_br)
    ax.set_ylim(min(0, per.to_numpy(float).min()) - 0.04 * top, top * (1.42 + 0.19 * len(con) + 0.08))
    ax.axhline(0, ls="-", lw=1, color="k")
    ax.set_xticks(range(K))
    ax.set_xticklabels([LAB[k] for k in X], rotation=25, ha="right", fontsize=fs_tick)
    ax.set_ylabel("replay rate: deviance explained\nlost on dropping (pp)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    D = compute()
    report(D)
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 13, "axes.labelsize": 13, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "axes.linewidth": 1.5, "legend.fontsize": 11,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    draw(ax, D, compact=False)
    fig.text(0.01, 0.01,
             f"n = {D['n_sess']} sessions, {D['n_animal']} mice, {D['n_win']:,} windows\n"
             "bars: full " + chr(8722) + " ablated GLM " + chr(177) + " 95% CI (cluster bootstrap over sessions); dots: sessions\n"
             "stars: mixed-model fixed-effect z; brackets: d2 vs each competitor, Wald contrast, Holm",
             ha="left", va="bottom", fontsize=8.5, color="0.3")
    plt.tight_layout(rect=(0, 0.10, 1, 1))
    assert_local_path(FIGDIR)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"fig2f_standalone_excised.{ext}")), dpi=300)
    plt.close(fig)
    print(f"figure -> {os.path.join(FIGDIR, 'fig2f_standalone_excised')}.png/.svg")


if __name__ == "__main__":
    main()
