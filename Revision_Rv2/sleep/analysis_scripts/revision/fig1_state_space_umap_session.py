"""
fig1_state_space_umap_session.py -- one-session / one-animal version of the window-feature state-space
UMAP: a second representation, beside the LFP-feature manifold of `fig1_lfp_manifold.py` (first
representation).

Inputs: the non-overlapping 30 s windows of `fig1_state_space_features.csv` (a window lies wholly inside one
state, so a session whose theta bouts are shorter than a window has no theta-wake windows). d2 (2024
Euclidean, AR order 20) is a label, not a feature; ripple rate is one of the features.

Cohort, missing values and geometry are those of the within-NREM test, `fig1_nrem_feature_gradient.py`:
Fig-1 sessions with SWR detection only (16 of 18; PPP20 day9 and PPP21 day8 have no ripple file),
ripple_amp_mean set to the session mean in zero-ripple windows, features z-scored per session over all
of the session's windows, and the UMAP fitted with Euclidean distance on the same z-scored features the
structure index (SI) is computed on. SI (PridaLab `structure_index`, n_bins 10, n_neighbors 20, all
states) is descriptive and lives in the FSET-D feature space, not on the 2-D embedding. d2 and the
features are autocorrelated in time, so no label-permutation null is quoted. The
within-NREM test against a time-preserving null is read from `fig1_nrem_feature_gradient.csv` when
present. The d2 colour scale is the cohort's 2-98 % range in every figure, so colours compare across
figures.

SESSION  one session ("PPP20_day5_CA1"), an animal ("PPP20"), or "all" (every cohort session in one
         embedding); with more than one session d2 is shown z-scored within session
FSET     10 (default), 9 (minus ripple_amp_mean), 8 (minus ripple rate and amplitude), 9emg (9 + LFP EMG)
T0, T1   optional time range (s, window centre)

Writes only to D:.
    -> reports/revision_figs/fig1_state_umap_<session or animal>[_<T0>-<T1>s]_<FSET>f.png / .svg
    -> data/revision/log/fig1_state_umap_<session or animal>[_<T0>-<T1>s]_<FSET>f.csv
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                        # noqa: E402
from fig1_state_space_umap import (UMAP_KW, STATE_COLOR, STATE_LABEL, LOGDIR, FIGDIR,   # noqa: E402
                                   zscore_per_session)
from fig1_nrem_feature_gradient import SETS, cohort_sessions, prepare, OUT as NREM_TEST   # noqa: E402
SI_TEST = os.path.join(LOGDIR, "fig1_si_sessions.csv")         # written by fig1_si_sessions.py
from fig1_lfp_manifold import SI_PATH                                 # noqa: E402

SESSION = os.environ.get("SESSION", "PPP20_day5_CA1")
FSET = os.environ.get("FSET", "10")
T0, T1 = float(os.environ.get("T0", "-inf")), float(os.environ.get("T1", "inf"))
UMAP_EUCLID = {**UMAP_KW, "metric": "euclidean"}
# 9 features + the sleep score's LFP-derived EMG (fig1_emg_lfp.py); theta is already in as theta_ratio,
# buzcode's theta power above 1/f
EXTRA_SETS = {"9emg": SETS["9"] + ["EMG_lfp"]}


def si(X, y, k=20):
    if SI_PATH not in sys.path:
        sys.path.insert(0, SI_PATH)
    from structure_index import compute_structure_index
    ok = np.isfinite(y) & np.isfinite(X).all(1)
    r = compute_structure_index(X[ok], y[ok].copy(), n_bins=10, n_neighbors=k, num_shuffles=0, verbose=False)
    return float(r[0])


def nrem_test_note(sessions):
    """the test result for these sessions, if it covers every one of them: with all 10 features, the per-session SI
    against its own time-preserving null (fig1_si_sessions.py, all states); otherwise the within-NREM test
    (fig1_nrem_feature_gradient.py) for this feature set. A test run on fewer sessions than the figure is not quoted."""
    if FSET == "10" and os.path.exists(SI_TEST):
        R = pd.read_csv(SI_TEST)
        R = R[R.session.isin(sessions)]
        if set(R.session) == set(sessions):
            return (f"\nper session vs its own time-shift null:\n{int((R.SI > R.null_p95).sum())}/{len(R)} sessions above"
                    if len(R) > 1 else f"\nvs its own time-shift null: 95th {R.null_p95.iloc[0]:.2f}, p = {R.p.iloc[0]:.3f}")
    if not os.path.exists(NREM_TEST):
        return ""
    R = pd.read_csv(NREM_TEST, dtype={"feature_set": str})
    R = R[(R.variant == "primary") & (R.feature_set == FSET) & R.session.isin(sessions)]
    if R.empty or set(R.session) != set(sessions):
        return ""
    if len(R) == 1:
        r = R.iloc[0]
        return (f"\nNREM vs time-shift null:\nsmoothness {r.smooth_effect:+.2f} (p = {r.p_smooth:.3f}), "
                f"SI {r.SI_excess:+.2f} (p = {r.p_SI:.3f})")
    return (f"\nNREM vs time-shift null, sessions p < .05:\nsmoothness {int((R.p_smooth < .05).sum())}/{len(R)}, "
            f"SI {int((R.p_SI < .05).sum())}/{len(R)}")


def main():
    feats = {**SETS, **EXTRA_SETS}[FSET]
    F = pd.read_csv(os.path.join(LOGDIR, "fig1_state_space_features.csv"))
    if "EMG_lfp" in feats:
        E = pd.read_csv(os.path.join(LOGDIR, "fig1_emg_lfp_windows.csv"))[["session", "center_s", "EMG_lfp"]]
        F = F.merge(E, on=["session", "center_s"], how="left")
    F["ripple_rate_hz"] = F.ripple_rate                               # prepare() z-scores the features in place
    Z = prepare(F, cohort_sessions(F))
    if "EMG_lfp" in feats:                                            # z-scored per session like the other features
        print(f"EMG_lfp missing in {int(Z.EMG_lfp.isna().sum())} windows (dropped)")
        Z = zscore_per_session(Z.dropna(subset=["EMG_lfp"]).reset_index(drop=True), ["EMG_lfp"])
    # several sessions in one picture: d2 z-scored within session, since raw session medians track unit yield
    # (fig1_state_space_d2z.py); one session: raw d2 on the 16-session 2-98 % range
    Z["d2_z"] = Z.groupby("session").d2.transform(lambda v: (v - v.mean()) / v.std(ddof=0))
    lo, hi = np.nanpercentile(Z.d2, [2, 98])
    sel = (np.ones(len(Z), bool) if SESSION == "all" else (Z.session == SESSION) if "_day" in SESSION
           else Z.session.str.startswith(SESSION + "_"))
    D = Z[sel & (Z.center_s >= T0) & (Z.center_s <= T1)].reset_index(drop=True)
    if D.empty:
        raise SystemExit(f"{SESSION}: not in the cohort (no SWR detection, or not a Fig-1 session)")
    multi = D.session.nunique() > 1
    d2_col, (lo, hi) = ("d2_z", (-2.0, 2.0)) if multi else ("d2", (lo, hi))
    import umap
    emb = umap.UMAP(**UMAP_EUCLID).fit_transform(D[feats].to_numpy(float))
    D["u1"], D["u2"] = emb[:, 0], emb[:, 1]
    tag = ((SESSION.replace("_CA1", "") if "_day" in SESSION else f"all{D.session.nunique()}" if SESSION == "all"
            else f"{SESSION}_{D.session.nunique()}days")
           + (f"_{int(T0)}-{int(T1)}s" if np.isfinite(T0) else "") + f"_{FSET}f")
    D.to_csv(assert_local_path(os.path.join(LOGDIR, f"fig1_state_umap_{tag}.csv")), index=False)
    res = {lab: si(D[feats].to_numpy(float), D[col].to_numpy(float)) for lab, col in (("d2", d2_col), ("ripple_rate", "ripple_rate_hz"))}
    print(f"{SESSION} ({FSET} features): {len(D)} windows from {sorted(D.session.unique())}, "
          f"states {D.state.value_counts().to_dict()}")
    for k, v in res.items():
        print(f"  SI {k:11s} in the {FSET}-D z-scored feature space (all states): {v:.3f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "svg.fonttype": "none"})
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.9))
    ms = 1.5 if len(D) > 5000 else 4 if len(D) > 1500 else 12
    for st in ("NREM", "nonTHETA", "REM", "THETA"):                  # NREM first: rare states drawn on top
        g = D[D.state == st]
        if len(g):
            ax[0].scatter(g.u1, g.u2, s=ms, c=STATE_COLOR[st], lw=0, label=f"{STATE_LABEL[st]} ({len(g)})")
    ax[0].legend(frameon=False, fontsize=7, markerscale=12 / ms if ms < 12 else 1); ax[0].set_title("a  state", loc="left")
    h = ax[1].scatter(D.u1, D.u2, s=ms, c=D[d2_col], cmap="viridis", vmin=lo, vmax=hi, lw=0)
    plt.colorbar(h, ax=ax[1], fraction=0.046,
                 label="d2 (2024 Euclidean), z within session" if multi else "d2 (2024 Euclidean)")
    ax[1].set_title(f"b  d2\nSI {res['d2']:.2f} (all states){nrem_test_note(set(D.session))}", loc="left", fontsize=8)
    h = ax[2].scatter(D.u1, D.u2, s=ms, c=D.ripple_rate_hz, cmap="magma", lw=0)
    plt.colorbar(h, ax=ax[2], fraction=0.046, label="SWR rate (Hz, per 30 s window)")
    role = "also a feature" if "ripple_rate" in feats else "not a feature here"
    ax[2].set_title(f"c  SWR rate ({role})\nSI {res['ripple_rate']:.2f} (all states)", loc="left", fontsize=8)
    for a in ax:
        a.set_xticks([]); a.set_yticks([]); a.set_xlabel("UMAP 1"); a.set_ylabel("UMAP 2")
    days = sorted(D.session.unique(), key=lambda x: int(x.split("_")[1][3:]))
    span = (f"{T0:.0f}-{T1:.0f} s" if np.isfinite(T0) else "whole session" if "_day" in SESSION
            else f"{D.session.nunique()} sessions, {D.session.str.split('_').str[0].nunique()} mice" if SESSION == "all"
            else "days " + ", ".join(x.split("_")[1][3:] for x in days))
    name = "All Fig-1 sessions with SWR detection" if SESSION == "all" else SESSION.replace("_CA1", "").replace("_", " ")
    fig.suptitle(f"{name} ({span}): UMAP of {len(feats)} window features "
                 "(z-scored within session, Euclidean), d2 (2024 Euclidean, p = 20) as a label", fontsize=9)
    colour = ("d2 z-scored within each session (raw session medians track unit yield), colour range -2..2"
              if multi else f"d2 colour range fixed to the cohort 2-98 % ({lo:.3f}-{hi:.3f}) in every single-session figure")
    fig.text(0.01, -0.02,
             f"Non-overlapping 30 s windows wholly inside one state. SI: PridaLab structure index in the {len(feats)}-D "
             "feature space (not on the 2-D embedding), descriptive; an NREM line is the time-preserving test of "
             f"fig1_nrem_feature_gradient.\n{colour}. Features: {', '.join(feats)}.", fontsize=6.5, va="top")
    fig.tight_layout()
    o = os.path.join(FIGDIR, f"fig1_state_umap_{tag}")
    for ext in ("png", "svg"):
        fig.savefig(assert_local_path(f"{o}.{ext}"), dpi=200, bbox_inches="tight")
    print("->", o + ".png")


if __name__ == "__main__":
    main()
