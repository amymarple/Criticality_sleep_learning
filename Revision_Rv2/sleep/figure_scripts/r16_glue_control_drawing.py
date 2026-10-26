"""Drawing functions from analysis_scripts/revision/ (r16_glue_control.py, r16_event_dissociation.py) used by the
figure notebooks. LOGDIR, FIGDIR: the folder of the notebook's tables, set by the notebook (plot() takes FIGDIR from
this module; the producer imports it from r16_block_level_dissociation). main() takes the two window tables the
figure draws.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

from final_figs import assert_local_path   # the producers' write guard; the figures are captured, not written

LOGDIR = None
FIGDIR = None


# --- r16_event_dissociation.py
def partial_spearman(a, b, C):
    ra, rb = stats.rankdata(a), stats.rankdata(b)
    A = np.column_stack([np.ones(len(ra))] + [stats.rankdata(c) for c in C.T])
    ea = ra - A @ np.linalg.lstsq(A, ra, rcond=None)[0]
    eb = rb - A @ np.linalg.lstsq(A, rb, rcond=None)[0]
    return float(stats.pearsonr(ea, eb)[0])


# --- r16_glue_control.py
def signed(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return dict(median=float(np.median(v)), n=int(v.size),
                pos=int((v > 0).sum()), neg=int((v < 0).sum()),
                p=float(stats.wilcoxon(v).pvalue) if v.size > 1 else np.nan)


# --- r16_glue_control.py
PER_SESSION = []      # (side, arm, target, session, rho) -- filled by corr_table, read by plot()


# --- r16_glue_control.py
def corr_table(D, arms, targets, side, min_n=30):
    """Per arm x target: within-session Spearman (and partial on the other target when two)."""
    out = []
    for arm, lab in arms:
        for tgt, tlab in targets:
            rho, prho = [], []
            other = [t for t, _ in targets if t != tgt]
            for s, h in D.groupby("session"):
                h = h.dropna(subset=[arm] + [t for t, _ in targets])
                if len(h) < min_n or h[arm].std() == 0:
                    continue
                rho.append(stats.spearmanr(h[arm], h[tgt]).correlation)
                PER_SESSION.append(dict(side=side, arm=arm, arm_label=lab, target=tgt,
                                        target_label=tlab, session=s, rho=rho[-1]))
                if len(other) == 1:
                    prho.append(partial_spearman(h[arm].to_numpy(), h[tgt].to_numpy(),
                                                 h[other].to_numpy()))
            r = signed(rho)
            pr = signed(prho) if prho else {}
            out.append(dict(side=side, arm=arm, arm_label=lab, target=tgt, target_label=tlab,
                            rho_median=r["median"], n_sessions=r["n"], pos=r["pos"], neg=r["neg"],
                            p=r["p"], partial_median=pr.get("median", np.nan),
                            partial_pos=pr.get("pos", np.nan), partial_neg=pr.get("neg", np.nan),
                            partial_p=pr.get("p", np.nan)))
    return pd.DataFrame(out)


# --- r16_glue_control.py
def show(T, title):
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")
    print(f"  {'arm':22s}{'target':18s}{'rho med':>9s}{'+/-/n':>10s}{'p':>9s}"
          f"{'partial':>9s}{'+/-':>8s}{'p':>9s}")
    for _, r in T.iterrows():
        pp = (f"{r.partial_median:9.3f}{int(r.partial_pos):4d}/{int(r.partial_neg):<3d}{r.partial_p:9.4f}"
              if np.isfinite(r.partial_median) else "")
        print(f"  {r.arm_label:22s}{r.target_label:18s}{r.rho_median:+9.3f}"
              f"{r.pos:4d}/{r.neg:<2d}/{r.n_sessions:<3d}{r.p:9.4f}{pp}")


# --- r16_glue_control.py
def plot(T, with_sham=False):
    """Three panels, per-session Spearman (dots), box over sessions, signed-rank stars, one
    box per arm: a: d2 vs ripple rate; b: d2 vs barrage rate (barrage cohort, 12 sessions);
    c: d2 vs replay probability (replay set, 8 sessions; the sham arm from the sham table).
    The sham arm sits in backup/; with_sham=True redraws the four-arm version into
    backup/r16_glue_control_with_sham."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    mpl.rcParams.update({"svg.fonttype": "none", "font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
                         "font.size": 11, "axes.linewidth": 1.4,
                         "xtick.major.width": 1.4, "ytick.major.width": 1.4})
    P = pd.DataFrame(PER_SESSION)
    COL = {"intact": "0.75", "glued": "#7b3294", "segments": "#1b9e77", "sham": "#fdb863"}
    KIND = {"d2_intact": "intact", "d2_glued": "glued", "d2_segments": "segments", "d2_sham": "sham",
            "d2_published": "intact", "ols_exc_glued_thr": "glued", "yw_exc_thr": "segments",
            "py_sham": "sham"}
    LAB = {"intact": "intact\nwindow", "glued": "excised,\nglued\n(current)",
           "segments": "excised,\nwithin-\nsegment", "sham": "sham-\nexcised,\nglued"}
    panels = [
        ("a  d2 vs ripple rate", "barrage cohort", "ripple_rate",
         ["d2_intact", "d2_glued", "d2_segments", "d2_sham"]),
        ("b  d2 vs barrage rate", "barrage cohort", "barrage_rate",
         ["d2_intact", "d2_glued", "d2_segments", "d2_sham"]),
        ("c  d2 vs replay probability", None, None, None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.6 if with_sham else 10.2, 4.9))
    for ax, (title, side, tgt, arms) in zip(axes, panels):
        if side is None:   # replay: three arms from yw_match, sham from the sham table
            Q = pd.concat([P[(P.side == "replay set (yw_match)") & (P.target == "replay_prob")
                             & P.arm.isin(["d2_published", "ols_exc_glued_thr", "yw_exc_thr"])],
                           P[(P.side == "replay set (sham)") & (P.target == "replay_frac")
                             & (P.arm == "py_sham")]], ignore_index=True)
            arms = ["d2_published", "ols_exc_glued_thr", "yw_exc_thr", "py_sham"]
            n_lab = "8 sessions"
        else:
            Q = P[(P.side == side) & (P.target == tgt) & P.arm.isin(arms)]
            n_lab = "12 sessions"
        if not with_sham:
            arms = [a for a in arms if KIND[a] != "sham"]
            Q = Q[Q.arm.isin(arms)]
        Q = Q.assign(kind=Q.arm.map(KIND))
        order = [KIND[a] for a in arms]
        sns.boxplot(data=Q, x="kind", y="rho", order=order, ax=ax, width=0.6, showfliers=False,
                    linewidth=1.8, palette=COL, hue="kind", legend=False)
        rng = np.random.default_rng(2)
        ymax = max(0.05, Q.rho.max()); ymin = min(-0.05, Q.rho.min())
        pad = 0.12 * (ymax - ymin)
        for i, k in enumerate(order):
            v = Q[Q.kind == k].rho.to_numpy()
            ax.plot(i + rng.uniform(-0.13, 0.13, v.size), v, "o", ms=4, color="k", alpha=0.6, mew=0)
            pw = stats.wilcoxon(v).pvalue if v.size > 1 else np.nan
            same = max(int((v > 0).sum()), int((v < 0).sum()))
            st = "***" if pw < .001 else "**" if pw < .01 else "*" if pw < .05 else "n.s."
            ax.text(i, ymax + 1.1 * pad, f"{same}/{v.size}\n{st}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold" if pw < .05 else "normal")
        ax.axhline(0, ls="--", lw=1.2, color="k")
        ax.set_ylim(ymin - pad, ymax + 3.6 * pad)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([LAB[k] for k in order], fontsize=9)
        ax.set_xlabel("")
        ax.set_ylabel("within-session Spearman \u03c1" if ax is axes[0] else "")
        ax.set_title(f"{title}\n{n_lab}", loc="left", fontsize=10.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    note = ("Gluing control for the event-excised d2 (2024 Euclidean, p = 20). glued = the production estimator "
            "(event bins dropped, runs >= 1 s glued,\none AR(20) across the joins); within-segment = autocovariance "
            "accumulated inside each event-free segment only, no lag product crosses a join (Yule-Walker).\n"
            "Dots: sessions; numbers: sessions sharing the sign; stars: signed-rank across sessions. "
            "a-b: barrage cohort, ripples and barrages excised. c: replay set, ripples excised.")
    if not with_sham:
        fig.text(0.01, 0.006, note, fontsize=7.8, color="0.3", ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        assert_local_path(FIGDIR)
        for ext in ("png", "svg"):
            fig.savefig(os.path.join(FIGDIR, f"r16_glue_control.{ext}"), dpi=300)
        plt.close(fig)
        print(f"figure -> {os.path.join(FIGDIR, 'r16_glue_control')}.png/.svg")
        return
    fig.text(0.01, 0.006,
             "Gluing control for the event-excised d2 (2024 Euclidean, p = 20). glued = the production estimator "
             "(event bins dropped, runs >= 1 s glued, one AR(20) across the joins); within-segment = autocovariance "
             "accumulated inside each\nevent-free segment only, no lag product crosses a join (Yule-Walker); sham = the "
             "same number and durations of intervals at random event-free positions, removed and glued the same way "
             "(5 draws averaged).\nDots: sessions; numbers: sessions sharing the sign; stars: signed-rank across sessions. "
             "a-b: barrage cohort, ripples and barrages excised. c: replay set, ripples excised (sham arm from the "
             "replay sham table, replay per SWR).",
             fontsize=7.8, color="0.3", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    out = os.path.join(FIGDIR, "backup", "r16_glue_control_with_sham")
    os.makedirs(assert_local_path(os.path.dirname(out)), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{out}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"figure -> {out}.png/.svg")


# --- r16_glue_control.py (main, with the two window tables as inputs: B, the barrage cohort per window with the
# intact, glued and within-segment d2; Y, the replay set per window with the same three arms)
def main(B, Y):
    """Within-session Spearman of each arm with the event rates (barrage cohort) and with replay probability
    (replay set), printed, then the figure."""
    print(f"barrage cohort: {len(B)} windows, {B.session.nunique()} sessions; median d2 intact "
          f"{B.d2_intact.median():.4f}, glued {B.d2_glued.median():.4f}, segments "
          f"{B.d2_segments.median():.4f}; pooled r(glued, segments) "
          f"{B[['d2_glued', 'd2_segments']].corr().iloc[0, 1]:.3f}")
    arms_b = [("d2_intact", "intact"), ("d2_glued", "excised, glued (current)"),
              ("d2_segments", "excised, within-segment")]
    tb = corr_table(B, arms_b, [("ripple_rate", "ripple rate"), ("barrage_rate", "barrage rate")],
                    "barrage cohort")
    show(tb, "BARRAGE COHORT: within-session Spearman of d2 with the two event rates (partial = on the other rate)")
    arms_y = [("d2_published", "intact (published)"), ("ols_exc_glued_thr", "excised, glued OLS"),
              ("yw_exc_thr", "excised, within-segment YW")]
    ty = corr_table(Y, arms_y, [("replay_prob", "replay probability"), ("ripple_rate", "ripple rate")],
                    "replay set (yw_match)")
    show(ty, "REPLAY SET (d2_yw_matlab_match.csv, 8 sessions): Spearman with replay probability and ripple rate")
    T = pd.concat([tb, ty], ignore_index=True)
    plot(T)
