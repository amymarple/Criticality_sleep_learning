"""
fig1_lfp_manifold.py -- network-state manifold built from LFP state features (the lab's pupil-paper
representation: Chang, Tang et al., Nature 637, 2025, Fig. 1d-e), with d2 as a LABEL on it, not a feature.

The state labels are scored from LFP features (SleepScoreMaster: broadband slow wave, theta, EMG). Here
the LFP features define the state space and d2 (2024 Euclidean, AR order 20) is projected onto it, as
pupil size was.

Inputs: the pupil project's own preprocessing output, read from the share --
`<day>.SleepStateLFPfeatures_ALLStates-EP<k>.LFP.mat`: 1-s samples, columns [time, delta (1-4 Hz),
theta (5-10 Hz), ripple-band, broadband slope ("SW"), EMG from LFP]. Exists for the pupil sessions only
(PPP4 d8-11/18, PPP7 d8/12/14/23, PPP8 d7/8).

Design (pupil Fig. 1d settings): UMAP on the five LFP features (time column excluded), 2 components,
n_neighbors 20, min_dist 0.6, cosine (umap-learn; the published embedding used the MATLAB toolbox and a
saved template, so the layout differs, not the inputs). Labels per 1-s sample: state (SleepState,
no guard), d2 of the nearest window centre of `<day>_CA1_D2_time_resolved*.mat` (30 s / 2 s; within 1 s,
else unlabelled -- windows exist only inside state bouts), SWR rate (ripple peaks in +-15 s).
Structure index (PridaLab `structure_index`, Sebastian et al.; not pip-installed here: put its
`structure_index.py` on SI_PATH) of each label on the five features z-scored over the epoch,
Euclidean, n_neighbors 20 (all states) / 15 (NREM, the package default), NUM_SHUFFLES shuffles.
Also, for the pupil label, the published inputs (all six columns, time included, unscaled) beside the
z-scored five-feature inputs used elsewhere here -- the time-column comparison.

GEOMETRY=unified (default): the UMAP is fitted on the same z-scored features the SI is computed on,
Euclidean, so picture and statistic share one geometry. GEOMETRY=published: raw features, cosine -- the
pupil Fig. 1d setting, kept for the reproduction (output suffix _published). In both, SI lives in the
5-D feature space, not on the 2-D embedding. Consecutive 1-s samples carry the label of the same 30 s /
2 s window and are not independent, so the time-preserving test is done on the 30 s window features
(fig1_nrem_feature_gradient.py) rather than label permutation.

Reads the share read-only; writes only to D:.
    -> reports/revision_figs/fig1_lfp_manifold_<session>.png / .svg
    -> data/revision/log/fig1_lfp_manifold_<session>.csv      (embedding + labels)
    -> data/revision/log/fig1_lfp_manifold_si.csv             (structure indices)
"""
from __future__ import annotations

import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402
from scipy.io import loadmat                                          # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from run_fig1_metrics import assert_local_path                        # noqa: E402

REPO = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
LOGDIR = os.path.join(REPO, "data", "revision", "log")
FIGDIR = os.path.join(REPO, "reports", "revision_figs")
SI_PATH = os.environ.get("SI_PATH", r"C:\Users\Cornell\AppData\Local\Temp\claude\d--Documents-GitHub-Sleep-criticaltiy"
                                    r"\8f4bceef-983a-47ce-80af-ba2c3ae418f3\scratchpad\si_vendor")
ANIMAL, DAY = os.environ.get("ANIMAL", "PPP7"), os.environ.get("DAY", "day12")
EPS = [int(e) for e in os.environ.get("EPS", os.environ.get("EP", "4")).split(",")]
# optional time window, e.g. the Fig. 1 example (PPP20 day 5, 5200-10000 s -- the same session as
# the d2 example)
T0, T1 = float(os.environ.get("T0", "-inf")), float(os.environ.get("T1", "inf"))
BP = rf"W:\data\PPP\{ANIMAL}\{DAY}"
# features: the pupil project's file on the share, else a local run of its (path-patched) MATLAB preprocessing
FEAT_LOCAL = os.path.join(REPO, "data", "revision", "lfp_features")
SESS = (f"{ANIMAL}_{DAY}_EP{EPS[0]}" if len(EPS) == 1 and not np.isfinite(T0)
        else f"{ANIMAL}_{DAY}_{int(T0)}-{int(T1)}s" if np.isfinite(T0) else f"{ANIMAL}_{DAY}_EP{'-'.join(map(str, EPS))}")
FEATS = ["delta", "theta", "ripple", "slope", "EMG"]
NUM_SHUFFLES = int(os.environ.get("NUM_SHUFFLES", "100"))
GEOMETRY = os.environ.get("GEOMETRY", "unified")
UMAP_KW = dict(n_components=2, n_neighbors=20, min_dist=0.6, metric="euclidean" if GEOMETRY == "unified" else "cosine",
               random_state=42)
SFX = "" if GEOMETRY == "unified" else f"_{GEOMETRY}"


def load_features():
    parts = []
    for ep in EPS:
        f = os.path.join(BP, f"{DAY}.SleepStateLFPfeatures_ALLStates-EP{ep}.LFP.mat")
        if not os.path.exists(f):
            f = os.path.join(FEAT_LOCAL, f"{ANIMAL}_{DAY}.SleepStateLFPfeatures_ALLStates-EP{ep}.LFP.mat")
        m = loadmat(f)
        P = pd.DataFrame(m["LFP_features"].astype(float), columns=["t"] + FEATS)
        P["pupil"] = m["pupil_label"].ravel().astype(float)
        P["epoch"] = ep
        parts.append(P)
    D = pd.concat(parts, ignore_index=True)
    D = D[(D.t >= T0) & (D.t <= T1)].dropna(subset=FEATS).reset_index(drop=True)
    return D


def label_states(D):
    from neuro_py.io import loading
    st = loading.load_SleepState_states(BP)
    D["state"] = "other"
    for key, lab in (("WAKEstate", "WAKE"), ("NREMstate", "NREM"), ("REMstate", "REM")):
        for a, b in np.atleast_2d(st.get(key, np.zeros((0, 2)))):
            D.loc[(D.t >= a) & (D.t <= b), "state"] = lab
    th = np.atleast_2d(st.get("THETA", np.zeros((0, 2))))
    D["theta_wake"] = False
    for a, b in th:
        D.loc[(D.t >= a) & (D.t <= b) & (D.state == "WAKE"), "theta_wake"] = True
    return D


def label_d2(D):
    # the file the Fig. 1 example draws (fig1c_example_traces.py) when it exists, else the time-resolved .mat
    f = os.path.join(BP, "complex_system", f"{DAY}_CA1_D2_time_resolved_error_bar_forplot.mat")
    if not os.path.exists(f):
        f = os.path.join(BP, "complex_system", f"{DAY}_CA1_D2_time_resolved.mat")
    m = loadmat(f, squeeze_me=True, struct_as_record=False, simplify_cells=True)
    W = pd.concat([pd.DataFrame({"c": np.atleast_1d(r["center"]).astype(float),
                                 "d2": np.atleast_1d(r["d2"]).astype(float)}) for r in m["results"]]).sort_values("c")
    i = np.clip(np.searchsorted(W.c.to_numpy(), D.t.to_numpy()), 1, len(W) - 1)
    c0, c1 = W.c.to_numpy()[i - 1], W.c.to_numpy()[i]
    j = np.where(np.abs(D.t - c0) <= np.abs(D.t - c1), i - 1, i)
    near = np.abs(W.c.to_numpy()[j] - D.t.to_numpy()) <= 1.0
    D["d2"] = np.where(near, W.d2.to_numpy()[j], np.nan)
    return D


def label_swr(D):
    from neuro_py.io import loading
    rip = loading.load_ripples_events(BP)
    pk = np.sort(np.asarray(rip["peaks"] if "peaks" in rip.columns else rip["start"], float))
    D["swr_rate"] = (np.searchsorted(pk, D.t + 15) - np.searchsorted(pk, D.t - 15)) / 30.0
    return D


def si(data, label, n_neighbors):
    sys.path.insert(0, SI_PATH)
    from structure_index import compute_structure_index
    ok = np.isfinite(label) & np.isfinite(data).all(1)
    r = compute_structure_index(data[ok], label[ok], n_bins=10, n_neighbors=n_neighbors,
                                num_shuffles=NUM_SHUFFLES, verbose=False)
    sh = np.asarray(r[3], float)
    return float(r[0]), float(np.percentile(sh, 95)), int(ok.sum())


def main():
    if os.environ.get("PLOT_ONLY"):                                  # redraw from the saved embedding + SI table
        D = pd.read_csv(os.path.join(LOGDIR, f"fig1_lfp_manifold_{SESS}{SFX}.csv"))
        rows = pd.read_csv(os.path.join(LOGDIR, "fig1_lfp_manifold_si.csv")).query("session == @SESS").to_dict("records")
        return plot(D, rows)
    D = label_swr(label_d2(label_states(load_features())))
    print(f"{SESS}: {len(D)} 1-s samples; states {D.state.value_counts().to_dict()}; "
          f"d2 labelled {int(D.d2.notna().sum())}")
    import umap
    Z = (D[FEATS] - D[FEATS].mean()) / D[FEATS].std()
    emb = umap.UMAP(**UMAP_KW).fit_transform((Z if GEOMETRY == "unified" else D[FEATS]).to_numpy())
    D["u1"], D["u2"] = emb[:, 0], emb[:, 1]
    D.to_csv(assert_local_path(os.path.join(LOGDIR, f"fig1_lfp_manifold_{SESS}{SFX}.csv")), index=False)

    rows = []
    nrem = (D.state == "NREM").to_numpy()
    has_pupil = bool(D.pupil.notna().any())
    for lab in ("d2", "swr_rate") + (("pupil",) if has_pupil else ()):
        for scope, mask, k in (("all states", np.ones(len(D), bool), 20), ("NREM", nrem, 15)):
            s, s95, n = si(Z.to_numpy()[mask], D[lab].to_numpy()[mask], k)
            rows.append(dict(session=SESS, label=lab, scope=scope, inputs="5 LFP features, z-scored",
                             SI=s, shuffle_p95=s95, n=n, n_neighbors=k))
            print(f"  SI {lab:9s} {scope:10s} z-scored 5 features: {s:.3f} (shuffle 95th {s95:.3f}), n = {n}")
    # the published inputs: all six columns (time first), unscaled -- the time-column check
    raw6 = D[["t"] + FEATS].to_numpy()
    for scope, mask, k in ((("all states", np.ones(len(D), bool), 20), ("NREM", nrem, 15)) if has_pupil else ()):
        s, s95, n = si(raw6[mask], D.pupil.to_numpy()[mask], k)
        rows.append(dict(session=SESS, label="pupil", scope=scope, inputs="6 columns incl. time, unscaled (published code)",
                         SI=s, shuffle_p95=s95, n=n, n_neighbors=k))
        print(f"  SI pupil     {scope:10s} published inputs (time + 5, raw): {s:.3f} (shuffle 95th {s95:.3f})")
    out = os.path.join(LOGDIR, "fig1_lfp_manifold_si.csv")
    old = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    if len(old):
        old = old[old.session != SESS]
    pd.concat([old, pd.DataFrame(rows)]).to_csv(assert_local_path(out), index=False)
    plot(D, rows)


def plot(D, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "svg.fonttype": "none"})
    col = {"NREM": "#7b3294", "REM": "#d7b5d8", "WAKE": "#fdb863", "other": "#dddddd"}
    S = pd.DataFrame(rows).set_index(["label", "scope", "inputs"])
    zin = "5 LFP features, z-scored"
    fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.8))               # panels: state, d2, SWR rate
    groups = [("transition", D.state == "other", "#dddddd"), ("wake, non-theta", (D.state == "WAKE") & ~D.theta_wake, "#fdb863"),
              ("wake, theta", (D.state == "WAKE") & D.theta_wake, "#d95f0e"), ("REM", D.state == "REM", col["REM"]),
              ("NREM", D.state == "NREM", col["NREM"])]
    for lab, msk, c in groups:                                       # one scatter per group: legend colours match
        ax[0].scatter(D.u1[msk], D.u2[msk], s=3, c=c, lw=0, label=lab)
    ax[0].legend(frameon=False, fontsize=7, markerscale=2.5, loc="best"); ax[0].set_title("a  state", loc="left")
    lo, hi = np.nanpercentile(D.d2, [2, 98])
    ax[1].scatter(D.u1, D.u2, s=3, c="#dddddd", lw=0)
    h = ax[1].scatter(D.u1, D.u2, s=3, c=D.d2, cmap="viridis", vmin=lo, vmax=hi, lw=0)
    plt.colorbar(h, ax=ax[1], fraction=0.046, label="d2 (2024 Euclidean)")
    r = S.loc[("d2", "all states", zin)]
    ax[1].set_title(f"b  d2\nSI {r.SI:.2f} (5-D feature space)", loc="left")
    h = ax[2].scatter(D.u1, D.u2, s=3, c=D.swr_rate, cmap="magma", lw=0)
    plt.colorbar(h, ax=ax[2], fraction=0.046, label="SWR rate (Hz, ±15 s)")
    r = S.loc[("swr_rate", "all states", zin)]
    ax[2].set_title(f"c  SWR rate\nSI {r.SI:.2f} (5-D feature space)", loc="left")
    for a in ax:
        a.set_xticks([]); a.set_yticks([]); a.set_xlabel("UMAP 1"); a.set_ylabel("UMAP 2")
    span = (f"{T0:.0f}-{T1:.0f} s" if np.isfinite(T0) else f"epoch {','.join(map(str, EPS))}")
    geo = ("z-scored features, Euclidean: the space SI is computed in" if GEOMETRY == "unified"
           else "raw features, cosine: pupil Fig. 1d setting")
    fig.suptitle(f"{ANIMAL} {DAY}, {span}: UMAP of the five LFP state features (delta, theta, ripple band, "
                 f"slope, EMG; {geo}), d2 (2024 Euclidean) as a label", fontsize=9)
    fig.text(0.01, -0.02, "SI: PridaLab structure index in the z-scored 5-D LFP feature space (Euclidean), not on the 2-D "
             "embedding. 1-s samples carry d2 of 30 s windows stepped 2 s (r = 0.96 at 2 s), so a label-permutation null "
             "is too loose and none is quoted;\nthe time-preserving test is on the 30 s window features "
             "(fig1_nrem_feature_gradient). d2 colour clipped to this span's 2-98 %.", fontsize=7, va="top")
    fig.tight_layout()
    o = os.path.join(FIGDIR, f"fig1_lfp_manifold_{SESS}{SFX}")
    fig.savefig(assert_local_path(o + ".png"), dpi=200, bbox_inches="tight"); fig.savefig(o + ".svg", bbox_inches="tight")
    print("->", o + ".png")


if __name__ == "__main__":
    main()
