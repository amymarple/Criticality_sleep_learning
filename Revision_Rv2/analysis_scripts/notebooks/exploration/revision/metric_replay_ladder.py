"""metric_replay_ladder.py -- the binomial ladder relating branching ratio and ISI entropy to
per-SWR replay odds, with each metric in d2's seat, on one common window set.

The model is the one used for d2: binomial GLM of replay_count / n_swr per window,
metric z-scored within session, covariates added one at a time, session as the unit.

Windows: Fig 1's non-overlapping NREM tiles, 8 decoded sessions (the table written by
excised_br_entropy.py: ripple_rate, replay_prob, MUA_mean, MUA_std and every metric in
full-window and ripple-excised form). n_swr = ripple_rate x 30 s, n_replay = replay_prob x n_swr
(both integers by construction of build_events). Windows with n_swr >= 1.

Metrics (the "d2 seat"):
    d2_excised            2024 Euclidean d2, ripple-excised (.mat)      -- the reference
    d2_published          2024 Euclidean d2, full window
    BR_pyr_k200           branching ratio, full window (Fig 1 pickle)
    BR_excised            branching ratio on the ripple-free sub-windows
    isi_entropy_pooled    pooled log-ISI entropy, full window
    isi_excised           pooled log-ISI entropy, ripple-free sub-windows
Ladder: M1 metric | M2 + MUA | M5 + MUA + SWR rate. b1 = log-odds of replay per +1
within-session SD. FR_STD=1 inserts a FR_std rung, EI=1 FR_std + E/I rungs, MUA_STD=1 an
alternate MUA_std ladder; BG=1 puts MUA and FR_std in the seat (bg_main). The seat-rotation figure
(d2 / MUA / SWR rate, and the joint model) is fig2f_seat_ladders.py, which reads the window
table this script writes.

    -> data/revision/log/metric_replay_ladder.csv
"""
from __future__ import annotations

import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                       # noqa: E402
from r35_binomial_replay import fit_binomial, pooled_fit, zs, WIN_S   # noqa: E402
from r35_glm_ablation import LOGDIR                                   # noqa: E402

WIN_CSV = os.path.join(LOGDIR, "excised_br_entropy_windows.csv")
OUT = os.path.join(LOGDIR, "metric_replay_ladder.csv")
METRICS = [("d2_excised", "d2, ripple-excised"), ("d2_published", "d2, full window"),
           ("BR_excised", "BR, ripple-excised"), ("BR_pyr_k200", "BR, full window"),
           ("isi_excised", "ISI entropy, ripple-excised"), ("isi_entropy_pooled", "ISI entropy, full window")]
# Covariate set = the submitted Fig 2F's (d2, MUA, FR_std), cumulative, then SWR rate. FR_std is
# the across-neuron SD of per-neuron rates in the window (generate_GLM.ipynb; computed here with
# fig2f_background_state_glm.window_fr_std on the full window), E/I is the Fig-1 pickle's
# EI_logratio. MUA_std remains available as an alternative rung (MUA_STD=1).
MODELS = [("M1", ["m_z"], "metric"),
          ("M2", ["m_z", "MUA_z"], "+ MUA"),
          ("M5", ["m_z", "MUA_z", "rate_z"], "+ SWR rate")]
if os.environ.get("FR_STD"):   # inserts the FR_std rung (forms a suppressor pair with MUA); FR_STD=1 enables it
    MODELS = [("M1", ["m_z"], "metric"),
              ("M2", ["m_z", "MUA_z"], "+ MUA"),
              ("M3", ["m_z", "MUA_z", "FR_std_z"], "+ FR_std"),
              ("M5", ["m_z", "MUA_z", "FR_std_z", "rate_z"], "+ SWR rate")]
if os.environ.get("EI"):   # inserts the E/I rung (absorbs part of d2's coefficient); EI=1 enables it
    MODELS = [("M1", ["m_z"], "metric"),
              ("M2", ["m_z", "MUA_z"], "+ MUA"),
              ("M3", ["m_z", "MUA_z", "FR_std_z"], "+ FR_std"),
              ("M4", ["m_z", "MUA_z", "FR_std_z", "EI_z"], "+ E/I"),
              ("M5", ["m_z", "MUA_z", "FR_std_z", "EI_z", "rate_z"], "+ SWR rate")]
if os.environ.get("MUA_STD"):
    MODELS = [("M1", ["m_z"], "metric"),
              ("M2", ["m_z", "MUA_z"], "+ MUA"),
              ("M3", ["m_z", "MUA_z", "MUA_std_z"], "+ MUA_std"),
              ("M5", ["m_z", "MUA_z", "MUA_std_z", "rate_z"], "+ SWR rate")]


def add_fr_std(W):
    """Across-neuron SD of per-neuron rate per window, all CA1 units."""
    from neuro_py.io import loading
    from r35_swr_property_control import SESSIONS
    from fig2f_background_state_glm import window_fr_std
    out = np.full(len(W), np.nan)
    for s, idx in W.groupby("session").groups.items():
        bp = SESSIONS[s]
        st, _ = loading.load_spikes(bp, brainRegion="CA1")
        units = [np.asarray(u, float) for u in np.asarray(st.data, dtype=object)]
        rip = loading.load_ripples_events(bp)
        S = window_fr_std(units, np.array(rip[["start", "stop"]], float), W.loc[idx, "center_s"].to_numpy(float))
        out[np.asarray([W.index.get_loc(i) for i in idx])] = S[:, 0]
        print(f"  {s:12s} FR_std computed for {len(idx)} windows", flush=True)
    W["FR_std"] = out
    return W


def bg_main():
    """BG=1: MUA and FR_std each in the metric seat. Rungs: alone | + the other background
    covariate | + ripple-excised d2 | + SWR rate. Reads the window table written by the main run.
        -> data/revision/log/metric_replay_ladder_bg.csv, reports/revision_figs/metric_replay_ladder_bg.png
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    W = pd.read_csv(os.path.join(LOGDIR, "metric_replay_ladder_windows.csv"))
    for c in ("MUA", "FR_std", "rate", "d2_excised"):
        W[f"{c}_z"] = W.groupby("session")[c].transform(zs)
    seats = [("MUA", "FR_std", "MUA (population rate)"), ("FR_std", "MUA", "FR_std (across-neuron SD)")]
    rows = []
    for m, other, lab in seats:
        models = [("M1", [f"{m}_z"], m), ("M2", [f"{m}_z", f"{other}_z"], f"+ {other}"),
                  ("M3", [f"{m}_z", f"{other}_z", "d2_excised_z"], "+ d2 (excised)"),
                  ("M5", [f"{m}_z", f"{other}_z", "d2_excised_z", "rate_z"], "+ SWR rate")]
        for tag, cols, mlab in models:
            for s, sub in W.groupby("session"):
                r = fit_binomial(sub, cols)
                if r is not None:
                    rows.append(dict(metric=m, metric_label=lab, model=tag, model_label=mlab, session=s, **r))
            pf = pooled_fit(W, cols)
            if pf is not None:
                rows.append(dict(metric=m, metric_label=lab, model=tag, model_label=mlab, session="__pooled__", **pf))
    R = pd.DataFrame(rows)
    R.to_csv(assert_local_path(os.path.join(LOGDIR, "metric_replay_ladder_bg.csv")), index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    box = ["#c03d3e", "#d4595a", "#e57373", "#ee8f8f"]
    for ax, (m, other, lab) in zip(axes, seats):
        G = R[(R.metric == m) & (R.session != "__pooled__")]
        tags = [t for t in ("M1", "M2", "M3", "M5")]
        data = [G[G.model == t].b1.to_numpy() for t in tags]
        labels = [G[G.model == t].model_label.iloc[0] for t in tags]
        print(f"\n{lab}: b1 per SD (sessions positive, signed-rank p)")
        bp = ax.boxplot(data, positions=range(4), widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color="#333333", lw=1.6), whiskerprops=dict(color="k"), capprops=dict(color="k"))
        for patch, c in zip(bp["boxes"], box):
            patch.set_facecolor(c); patch.set_edgecolor("k"); patch.set_alpha(0.9)
        for i, b in enumerate(data):
            ax.scatter(np.random.default_rng(0).uniform(-0.15, 0.15, len(b)) + i, b, s=9, color="#444444", zorder=3)
            p = stats.wilcoxon(b).pvalue
            star = "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            ax.text(i, max(b) + 0.01, star, ha="center", va="bottom", fontsize=8, fontweight="bold")
            pooled = R[(R.metric == m) & (R.model == tags[i]) & (R.session == "__pooled__")]
            print(f"  {labels[i]:18s} {np.median(b):+.3f} ({int((b > 0).sum())}/{len(b)}, p={p:.3f})"
                  f"   pooled b={pooled.b1.iloc[0]:+.3f} p={pooled.p.iloc[0]:.2g}")
        ax.axhline(0, color="k", ls="--", lw=0.8)
        ax.set_xticks(range(4)); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(lab, fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("log-odds of replay per SD\n(binomial, per session)", fontsize=8)
    fig.suptitle("Background covariates in the metric seat -- 9 sessions, decoded NREM tiles", fontsize=9)
    fig.tight_layout()
    out = os.path.join(_HERE, "..", "..", "..", "reports", "revision_figs", "metric_replay_ladder_bg.png")
    fig.savefig(assert_local_path(os.path.abspath(out)), dpi=200)
    print("  ->", os.path.abspath(out))


def main():
    if os.environ.get("BG"):
        return bg_main()
    W = pd.read_csv(WIN_CSV)
    W["n_swr"] = (W["ripple_rate"] * WIN_S).round().astype(int)
    W = W[(W["n_swr"] >= 1) & W["replay_prob"].notna()].copy()   # decoded time only (NaN outside)
    W["n_replay"] = np.minimum((W["replay_prob"] * W["n_swr"]).round().astype(int), W["n_swr"])
    W["rate"] = W["n_swr"] / WIN_S
    W["MUA"] = W["MUA_mean"]
    W["EI"] = W["EI_logratio"]
    W = W.reset_index(drop=True)
    if not os.environ.get("MUA_STD"):
        W = add_fr_std(W)
        W.to_csv(assert_local_path(os.path.join(LOGDIR, "metric_replay_ladder_windows.csv")), index=False)
    for c in ("MUA", "MUA_std", "rate", "EI") + (("FR_std",) if "FR_std" in W else ()):
        W[f"{c}_z"] = W.groupby("session")[c].transform(zs)
    rows = []
    for m, lab in METRICS:
        D = W.dropna(subset=[m]).copy()
        D["m_z"] = D.groupby("session")[m].transform(zs)
        for tag, cols, mlab in MODELS:
            for s, sub in D.groupby("session"):
                r = fit_binomial(sub, cols)
                if r is not None:
                    rows.append(dict(metric=m, metric_label=lab, model=tag, model_label=mlab, session=s, **r))
            pf = pooled_fit(D, cols)
            if pf is not None:
                rows.append(dict(metric=m, metric_label=lab, model=tag, model_label=mlab, session="__pooled__", **pf))
    R = pd.DataFrame(rows)
    assert_local_path(LOGDIR)
    R.to_csv(assert_local_path(OUT), index=False)
    print(f"{W.session.nunique()} sessions, {len(W)} windows with >= 1 SWR, {int(W.n_swr.sum())} SWRs, "
          f"{W.n_replay.sum() / W.n_swr.sum():.1%} replay\n")
    print(f"{'metric':30s}" + "".join(f"{t + ': ' + l:>26s}" for t, _, l in MODELS))
    for m, lab in METRICS:
        cells = []
        for tag, _, _ in MODELS:
            P = R[(R.metric == m) & (R.model == tag) & (R.session != "__pooled__")]
            b = P.b1.to_numpy()
            p = stats.wilcoxon(b).pvalue if len(b) > 1 else np.nan
            cells.append(f"{np.median(b):+.3f} ({int((b > 0).sum())}/{len(b)}, p={p:.3f})")
        print(f"{lab:30s}" + "".join(f"{c:>26s}" for c in cells))
    print("\npooled (session FE, cluster SE) p at M5: " + "  ".join(
        f"{lab} {R[(R.metric == m) & (R.model == 'M5') & (R.session == '__pooled__')].p.iloc[0]:.2g}"
        for m, lab in METRICS))


if __name__ == "__main__":
    main()
