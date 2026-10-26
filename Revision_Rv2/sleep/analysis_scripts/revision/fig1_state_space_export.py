"""
fig1_state_space_export.py -- feature export for the Chang-2025-style state-space figure.

Merges the base metric pickles with the drop-in columns and writes ONE flat CSV
of per-window fingerprint features + overlay values, so the UMAP step can run
in a separate conda env (scientific-skills / dyngdim / can_field).

Design copied from the lab's own published paper (Chang et al., Nature 637, 2025,
Fig. 1d-e): 2-D embedding for visualization only; quantification happens in the
original high-dimensional space against label shuffles, in the UMAP step.

FEATURES = the physiological fingerprint (observables only). d2/BR/tau are
exported as OVERLAY columns, not embedding features -- they color the map the
way Chang et al. colored theirs with pupil size.

    C:/Users/Cornell/.conda/envs/criticality/python.exe fig1_state_space_export.py
    -> data/revision/log/fig1_state_space_features.csv
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aggregate_fig1_metrics import (  # noqa: E402
    assert_local_path, load_all, merge_dropin, OUTDIR,
)

KEYS = ["session", "animal", "state", "center_s", "nonoverlap"]
# CURATED: one column per metric family, so no family (especially firing rate)
# dominates the embedding distance. Dropped as redundant: MUA_std (~MUA_mean for
# counts), E_rate/I_rate (rate again + EI_logratio is derived from them),
# synchrony_raw_mean (n-dependent variant of sub81), isi_entropy_perunit_mean.
FEATURES = [
    "MUA_mean", "MUA_fano",            # rate + rate-normalized fluctuation
    "EI_logratio",
    "synchrony_sub81_mean",
    "isi_entropy_pooled",
    "psd_slope_20_45", "sw_broadband",
    "theta_ratio",                     # buzcode theta above 1/f (fig1_theta_ratio.py)
    # ripple features drop PPP20_day9 and PPP21_day8 from the map -- 
    # they have no *.ripples.events.mat on the share.
    "ripple_rate", "ripple_amp_mean",
]
OVERLAYS = ["d2", "BR_pyr_k200", "tau_ms_pyr_k200"]   # d2 = 2024 Euclidean


def main():
    W = load_all()
    W = merge_dropin(W)
    th_csv = os.path.join(OUTDIR, "fig1_theta_ratio_windows.csv")
    if os.path.exists(th_csv):
        n0 = len(W)
        TH = pd.read_csv(th_csv)[["session", "state", "center_s", "theta_ratio"]]
        W = W.merge(TH, on=["session", "state", "center_s"], how="left")
        assert len(W) == n0, "theta join changed the row count"
        print(f"theta_ratio joined: {W.theta_ratio.notna().sum()}/{n0} windows")
    else:
        print("[!] fig1_theta_ratio_windows.csv missing -- run fig1_theta_ratio.py")
    cols = KEYS + [c for c in FEATURES + OVERLAYS if c in W.columns]
    missing = [c for c in FEATURES + OVERLAYS if c not in W.columns]
    if missing:
        print(f"[!] missing columns (run the drop-in batch?): {missing}")
    out = W[cols]
    dst = assert_local_path(os.path.join(OUTDIR, "fig1_state_space_features.csv"))
    out.to_csv(dst, index=False)
    n_full = out[[c for c in FEATURES if c in out.columns]].notna().all(axis=1).sum()
    print(f"wrote {len(out)} windows ({n_full} with complete features) -> {dst}")
    print(out.groupby('state').size().to_string())


if __name__ == "__main__":
    main()
