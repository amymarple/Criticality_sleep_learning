"""
d2_composition_glm.py -- flipped GLM: d2 as the target, state metrics (incl. synchrony),
SWR activity and replay content as predictors, Shapley attribution.

The d2 -> replay GLM has two properties that this design changes: d2's per-window
error attenuates its coefficient as a predictor, and its effect size is small. With
d2 as the target, measurement error in the target adds residual noise without biasing
coefficients, and firing rate, SWR activity and background-regime metrics are
compared directly as predictors of d2.

r35_glm_ablation.py / r35_background_vs_ripple.py already run this ablation with SWR
rate, ripple amplitude, participating cells, MUA and replay probability as predictors.
This script keeps those groups and adds the remaining Fig-1 state metrics, synchrony
above all.

Data: Fig 1's validated per-window metrics (data/revision/<sess>_CA1_fig1_metrics_nonoverlap.pkl,
fig1_metrics.py): NREM, non-overlapping 30 s windows. Predictor groups:
    rate        MUA_mean, MUA_fano                        (40 ms bins)
    variability MUA_std
    E/I         EI_logratio
    synchrony   synchrony_raw_mean/std, synchrony_sub_mean/std   (50 ms, raw and 81-unit subsampled)
    ISI         isi_entropy_pooled            (the branching ratio is not a predictor: it is
                                               itself a criticality metric)
    SWR         ripple_rate, ripple_amp, n_cells          (per window from r35_swr_property_control.build_events)
    replay      replay_prob                                (same source; 9 decoded sessions only)
Targets: d2 (2024 Euclidean, p=20) ripple-excised (`*_rippleablate.mat`, 10 of the 18
Fig-1 sessions have it) and published (all 18, already joined in the pickles).

Model: per session, OLS of the target on within-session z-scored predictors, 5 contiguous
time-block folds, held-out R^2 (as r35_glm_ablation.cv_r2). Attribution = Shapley value
of each predictor over all 2^k subsets (k <= 14 -> <= 16 384 refits per session),
summed by group; leave-one-group-out dR^2 is reported alongside. Session is the unit.
Ceiling: lambda = reliability of the target from its per-window error (published:
d2_err in the pickle; excised: a fixed d2_err/d2 ratio of 0.216) -- R^2 cannot exceed it.

Reads the shares read-only; writes only to D:.
    -> data/revision/log/d2_composition_glm.csv           (per target x tier x session x predictor)
    -> data/revision/log/d2_composition_glm_groups.csv    (per target x tier x session x group)
    -> reports/revision_figs/d2_composition_{excised,published}.png/.svg
"""
from __future__ import annotations

import itertools
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
from run_fig1_metrics import FIG1_SESSIONS, session_id, assert_local_path      # noqa: E402
from r35_glm_ablation import LOGDIR, FIGDIR, WIN_S, N_FOLD                       # noqa: E402
from r35_background_vs_ripple import load_d2_mat                                 # noqa: E402
from r35_swr_property_control import SESSION_TEMPLATE, window_rate, window_mean  # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
DATA = os.path.join(REPO, "data", "revision")
ERR_RATIO_EXCISED = 0.216
JOIN_TOL_S = 1.0
GROUPS = {
    "rate":        ["MUA_mean", "MUA_fano"],
    "variability": ["MUA_std"],
    "E/I":         ["EI_logratio"],
    "synchrony":   ["synchrony_raw_mean", "synchrony_raw_std", "synchrony_sub_mean", "synchrony_sub_std"],
    "ISI entropy": ["isi_entropy_pooled"],
    "SWR":         ["ripple_rate", "ripple_amp", "n_cells"],
    "replay":      ["replay_prob"],
}
STATE_GROUPS = ["rate", "variability", "E/I", "synchrony", "ISI entropy"]
ALL_GROUPS = STATE_GROUPS + ["SWR", "replay"]
OUT_P = os.path.join(LOGDIR, "d2_composition_glm.csv")
OUT_G = os.path.join(LOGDIR, "d2_composition_glm_groups.csv")


# --------------------------------------------------------------------------- data
def fig2_basepaths():
    """Fig 1's 18 basepaths plus every replay-decoded session (SESSION_TEMPLATE) that has
    a fig1_metrics pickle but is not in the Fig-1 list (PPP7_day8), so the Fig 2 analyses
    run on all 9 decoded sessions."""
    from r35_swr_property_control import SESSIONS as _S
    bps = [b if isinstance(b, str) else b[0] for b in FIG1_SESSIONS]
    have = set()
    for b in bps:
        sid = session_id(b)
        have.add("_".join(sid) if isinstance(sid, (tuple, list)) else str(sid))
    for sess in SESSION_TEMPLATE:
        if sess not in have and os.path.exists(os.path.join(DATA, f"{sess}_CA1_fig1_metrics_nonoverlap.pkl")):
            bps.append(_S[sess])
    return bps


def load_session(bp):
    sid = session_id(bp)
    sid = "_".join(sid) if isinstance(sid, (tuple, list)) else str(sid)
    f = os.path.join(DATA, f"{sid}_CA1_fig1_metrics_nonoverlap.pkl")
    if not os.path.exists(f):
        return sid, None
    W = pd.read_pickle(f)["windows"]
    W = W[(W["state"] == "NREM") & (W["nonoverlap"].astype(bool))].copy()
    W = W.sort_values("center_s").reset_index(drop=True)
    W["d2_published"] = W["d2"].astype(float)
    W["d2_published_err"] = W["d2_err"].astype(float)
    m = load_d2_mat(bp, "_rippleablate")
    if m is not None and len(m):
        j = pd.merge_asof(W[["center_s"]], m.rename(columns={"center": "center_s"})
                          .sort_values("center_s")[["center_s", "d2_x"]],
                          on="center_s", direction="nearest", tolerance=JOIN_TOL_S)
        W["d2_excised"] = j["d2_x"].to_numpy(float)
    else:
        W["d2_excised"] = np.nan
    W["d2_excised_err"] = ERR_RATIO_EXCISED * W["d2_excised"]
    return sid, W


def add_swr(W, sid, bp):
    """ripple_rate / ripple_amp / n_cells / replay_prob per window, from the same
    per-ripple table used elsewhere in this directory (NREM-clipped ripples, session
    template)."""
    from r35_swr_property_control import build_events
    R = build_events(sid, bp, verbose=False)
    iv = R[["start", "stop"]].to_numpy(float)
    c = W["center_s"].to_numpy(float)
    W["ripple_rate"] = window_rate(iv, c)
    W["ripple_amp"] = window_mean(iv, R["amplitude"].to_numpy(float), c)
    W["n_cells"] = window_mean(iv, R["n_cells"].to_numpy(float), c)
    W["ripple_dur"] = window_mean(iv, R["duration"].to_numpy(float), c)
    W["replay_prob"] = window_mean(iv, R["is_replay"].to_numpy(float), c)
    # Replay labels exist only inside the decoded sleep epochs. Some sessions' replay
    # files cover only part of the sleep epochs (e.g. PPP7_day8 / PPP8_day8 / PPP21_day4
    # cover 2 of 3), so outside the span of the decoded candidate events the replay
    # probability is undefined, not zero.
    from r35_swr_property_control import load_replay_events
    E = load_replay_events(sid)
    inside = np.zeros(len(W), bool)
    for _, r in E.groupby("ep").agg(a=("start", "min"), b=("stop", "max")).iterrows():
        inside |= (c >= r.a - WIN_S / 2) & (c <= r.b + WIN_S / 2)
    W["decoded"] = inside
    W.loc[~inside, "replay_prob"] = np.nan
    return W


# --------------------------------------------------------------------------- model
def cv_r2_cols(X, y, cols, n_fold=N_FOLD):
    """Block-CV R^2 of OLS on the given predictor columns; empty -> 0."""
    if not cols:
        return 0.0
    n = len(y)
    b = np.linspace(0, n, n_fold + 1).astype(int)
    pred = np.full(n, np.nan)
    Xs = X[:, cols]
    for k in range(n_fold):
        te = np.zeros(n, bool)
        te[b[k]:b[k + 1]] = True
        tr = ~te
        mu, sd = Xs[tr].mean(0), Xs[tr].std(0)
        sd[sd == 0] = 1.0
        A = np.column_stack([np.ones(tr.sum()), (Xs[tr] - mu) / sd])
        beta = np.linalg.lstsq(A, y[tr], rcond=None)[0]
        pred[te] = np.column_stack([np.ones(te.sum()), (Xs[te] - mu) / sd]) @ beta
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan


def shapley_r2(X, y, names):
    from math import comb
    p = len(names)
    gains = {}
    for r in range(p + 1):
        for S in itertools.combinations(range(p), r):
            gains[S] = cv_r2_cols(X, y, list(S))
    phi = np.zeros(p)
    for j in range(p):
        others = [i for i in range(p) if i != j]
        for r in range(p):
            for S in itertools.combinations(others, r):
                Sj = tuple(sorted(S + (j,)))
                phi[j] += (gains[Sj] - gains[tuple(S)]) / (p * comb(p - 1, r))
    full = gains[tuple(range(p))]
    return phi, full, gains


def run(W, target, groups):
    preds = [c for g in groups for c in GROUPS[g]]
    sub = W[[target, f"{target}_err"] + preds].replace([np.inf, -np.inf], np.nan).dropna()
    if len(sub) < 40:
        return None, None
    X = sub[preds].to_numpy(float)
    y = sub[target].to_numpy(float)
    phi, full, gains = shapley_r2(X, y, preds)
    lam = max(1 - np.mean(sub[f"{target}_err"] ** 2) / y.var(ddof=1), 0.0)
    prow = [dict(predictor=c, group=g, shapley=phi[i], r2_full=full, lam=lam, n=len(sub))
            for i, (c, g) in enumerate((c, g) for g in groups for c in GROUPS[g])]
    grow = []
    allidx = tuple(range(len(preds)))
    for g in groups:
        idx = [preds.index(c) for c in GROUPS[g]]
        rest = tuple(i for i in allidx if i not in idx)
        grow.append(dict(group=g, shapley=float(sum(phi[i] for i in idx)),
                         logo_dr2=full - gains[rest], r2_full=full, lam=lam, n=len(sub)))
    return prow, grow


def main():
    prows, grows = [], []
    for bp in fig2_basepaths():
        sid, W = load_session(bp)
        if W is None:
            print(f"  [skip] {sid}: no fig1 pickle")
            continue
        has_ex = W["d2_excised"].notna().mean() > 0.5
        has_swr = sid in SESSION_TEMPLATE
        if has_swr:
            W = add_swr(W, sid, bp)
        print(f"  {sid:12s} {len(W):4d} NREM windows | excised d2 {'yes' if has_ex else 'no ':3s} | SWR/replay {'yes' if has_swr else 'no'}", flush=True)
        for target in ("d2_excised", "d2_published"):
            if target == "d2_excised" and not has_ex:
                continue
            for tier, groups in (("state", STATE_GROUPS), ("state+SWR+replay", ALL_GROUPS)):
                if tier != "state" and not has_swr:
                    continue
                pr, gr = run(W, target, groups)
                if pr is None:
                    continue
                prows += [dict(session=sid, target=target, tier=tier, **r) for r in pr]
                grows += [dict(session=sid, target=target, tier=tier, **r) for r in gr]
    P, G = pd.DataFrame(prows), pd.DataFrame(grows)
    assert_local_path(LOGDIR)
    P.to_csv(assert_local_path(OUT_P), index=False)
    G.to_csv(assert_local_path(OUT_G), index=False)
    print(f"\nwrote {OUT_P}\nwrote {OUT_G}")
    report(P, G)
    for target in ("d2_excised", "d2_published"):
        plot(G, target)


def _stars(p):
    return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."


def report(P, G):
    for target in ("d2_excised", "d2_published"):
        for tier in ("state", "state+SWR+replay"):
            g = G[(G.target == target) & (G.tier == tier)]
            if not len(g):
                continue
            ns = g.session.nunique()
            full = g.groupby("session").r2_full.first()
            lam = g.groupby("session").lam.first()
            print(f"\n=== TARGET {target} | predictors: {tier} | N = {ns} sessions | held-out R^2 "
                  f"median {full.median():.3f} (ceiling lambda {lam.median():.2f}, i.e. {full.median()/lam.median():.0%} of explainable) ===")
            print(f"  {'group':12s}{'Shapley R2':>11s}{'pos':>6s}{'p':>8s}{'share':>8s} | {'leave-group-out dR2':>20s}{'pos':>6s}{'p':>8s}")
            for grp in (STATE_GROUPS + ["SWR", "replay"]):
                q = g[g.group == grp]
                if not len(q):
                    continue
                v, d = q.shapley.to_numpy(), q.logo_dr2.to_numpy()
                share = (q.shapley / q.r2_full).median() if q.r2_full.median() > 0.01 else np.nan
                share_s = f"{share:8.0%}" if np.isfinite(share) else f"{'--':>8s}"
                print(f"  {grp:12s}{np.median(v):+11.3f}{int((v>0).sum()):3d}/{len(v):<2d}{stats.wilcoxon(v).pvalue:8.3f}{share_s} | "
                      f"{np.median(d):+20.3f}{int((d>0).sum()):3d}/{len(d):<2d}{stats.wilcoxon(d).pvalue:8.3f}")
            p = P[(P.target == target) & (P.tier == tier)]
            print("  per predictor (median Shapley R^2): " + "  ".join(
                f"{c} {p[p.predictor == c].shapley.median():+.3f}" for c in p.predictor.unique()))


def plot(G, target):
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 14,
                         "ytick.labelsize": 14, "axes.linewidth": 1.5,
                         "xtick.major.width": 1.5, "ytick.major.width": 1.5})
    tier = "state+SWR+replay"
    g = G[(G.target == target) & (G.tier == tier)]
    if not len(g):
        return
    order = [x for x in ALL_GROUPS if x in set(g.group)]
    long = g[["group", "shapley"]].rename(columns={"group": "epoch", "shapley": "value"})
    long["epoch"] = pd.Categorical(long["epoch"], categories=order, ordered=True)
    pal = dict(zip(order, sns.color_palette("Reds_r", len(order) + 2)[:len(order)]))
    fig, ax = plt.subplots(figsize=(5.2, 4))
    sns.boxplot(data=long, x="epoch", y="value", order=order, ax=ax, width=0.6, showfliers=False,
                palette=pal, linewidth=2, hue="epoch", legend=False)
    rng = np.random.default_rng(0)
    yr = long.value.max() - long.value.min()
    for i, grp in enumerate(order):
        v = g[g.group == grp].shapley.to_numpy()
        ax.plot(i + rng.uniform(-0.12, 0.12, v.size), v, "o", ms=3.5, color="k", alpha=0.6, mew=0, zorder=3)
        # one-sided: a star marks a group that helps held-out prediction (Fig2_F_GLM.ipynb's
        # test). A significantly negative Shapley value (noise under block CV) gets no star.
        ax.text(i, v.max() + 0.04 * yr, _stars(stats.wilcoxon(v, alternative="greater").pvalue),
                ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.axhline(0, ls="--", lw=1, color="k")
    ax.set_ylabel("Shapley share of held-out R$^2$\n(d2 as target, per session)")
    ax.set_xlabel("")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=35, ha="right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    assert_local_path(FIGDIR)
    os.makedirs(FIGDIR, exist_ok=True)
    tag = target.replace("d2_", "")
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(os.path.join(FIGDIR, f"d2_composition_{tag}.{ext}")), dpi=300)
    plt.close(fig)
    print(f"  -> {FIGDIR}\\d2_composition_{tag}.png/.svg")


if __name__ == "__main__":
    if os.environ.get("REPORT_ONLY"):
        P, G = pd.read_csv(OUT_P), pd.read_csv(OUT_G)
        report(P, G)
        for t in ("d2_excised", "d2_published"):
            plot(G, t)
    else:
        main()
