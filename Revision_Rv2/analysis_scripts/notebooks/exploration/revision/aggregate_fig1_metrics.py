"""
aggregate_fig1_metrics.py — group-level tables from the per-session revision pickles.

    C:/Users/Cornell/.conda/envs/criticality/python.exe aggregate_fig1_metrics.py

Reads data/revision/*_fig1_metrics_nonoverlap.pkl and writes:
    data/revision/log/fig1_session_medians.csv    one row per (session, state)
    data/revision/log/fig1_group_summary.csv      one row per (state, metric)
    data/revision/log/fig1_kmax_sweep.csv         per-state m at each k_max
    data/revision/log/fig1_d2_vs_br.csv           within-state d2~BR correlation

**Session is the unit of analysis.** Windows within a session are neither independent
nor equal in number across sessions, so every group statistic is computed over
per-session medians.
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fig1_metrics import BIN_BR_S, K_MAX_SWEEP, STATES  # noqa: E402

REPO = r"D:\Documents\GitHub\Sleep_criticaltiy"
INDIR = os.path.join(REPO, "data", "revision")
OUTDIR = os.path.join(INDIR, "log")

POPS = ("pyr", "all")          # pyramidal is the primary population
BR_PRIMARY = "BR_pyr_k200"
METRICS = [
    "d2", "BR_pyr_k200", "BR_all_k200", "tau_ms_pyr_k200", "tau_ms_all_k200",
    "BR_pyr_r1", "BR_all_r1",
    "MUA_mean", "MUA_std", "MUA_fano",
    "E_rate", "I_rate", "EI_logratio",
    "synchrony_raw_mean", "synchrony_sub_mean",
    "isi_entropy_pooled", "isi_entropy_perunit_mean",
    # drop-in columns (fig1_dropin_metrics.py) — merged on (session, state, center_s)
    # psd_slope_20_45 is the primary range
    "synchrony_sub81_mean", "sw_broadband", "ripple_rate", "ripple_amp_mean",
    "psd_slope_20_45", "psd_slope_30_45", "psd_slope_30_55", "psd_slope_65_90",
]
KCOLS = [f"BR_{p}_k{k}" for p in POPS for k in K_MAX_SWEEP]


def assert_local_path(path):
    p = os.path.abspath(str(path))
    if p.startswith("\\\\") or p.startswith("//"):
        raise PermissionError(f"refusing to write to UNC path: {p}")
    if os.path.splitdrive(p)[0].rstrip(":").upper() not in {"C", "D"}:
        raise PermissionError(f"refusing to write outside C:/D:: {p}")
    return p


def load_all(pattern="*_fig1_metrics_nonoverlap.pkl"):
    out = []
    for f in sorted(glob.glob(os.path.join(INDIR, pattern))):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        m = d["meta"]
        w = d["windows"].copy()
        w["session"] = f"{m['animal_name']}_{m['day']}_{m['region']}"
        w["animal"] = m["animal_name"]
        w["n_E_units"] = m["n_E_units"]
        out.append(w)
        print(f"  {os.path.basename(f):55s} {len(w):5d} windows  "
              f"d2 join {m.get('d2_join_rate', 0):.0%}")
    if not out:
        raise SystemExit(f"no pickles matching {pattern} in {INDIR}")
    return pd.concat(out, ignore_index=True)


def merge_dropin(W, pattern="*_fig1_dropin_nonoverlap.pkl"):
    """Left-merge the drop-in columns (fig1_dropin_metrics.py) onto the base
    windows, keyed on (session, state, center_s). No drop-in files -> W unchanged."""
    parts = []
    for f in sorted(glob.glob(os.path.join(INDIR, pattern))):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        m = d["meta"]
        w = d["windows"].copy()
        w["session"] = f"{m['animal_name']}_{m['day']}_{m['region']}"
        parts.append(w)
    if not parts:
        print("  (no drop-in pickles found — sw_broadband/psd_slope columns absent)")
        return W
    D = pd.concat(parts, ignore_index=True)
    keep = [c for c in D.columns
            if c not in ("state", "iv_idx", "w_idx", "win_start", "win_stop",
                         "center_s", "nonoverlap", "session")]
    for df in (W, D):
        df["_ckey"] = df["center_s"].round(3)
    n0 = len(W)
    W = W.merge(D[["session", "state", "_ckey"] + keep],
                on=["session", "state", "_ckey"], how="left")
    W = W.drop(columns="_ckey")
    assert len(W) == n0, f"drop-in merge changed row count {n0} -> {len(W)}"
    n_hit = W["sw_broadband"].notna().sum() if "sw_broadband" in W.columns else 0
    print(f"  drop-in merge: {len(parts)} sessions, "
          f"{n_hit}/{n0} windows with sw_broadband")
    return W


def iqr_str(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "n/a"
    return "%.3f [%.3f-%.3f]" % (np.median(v), *np.percentile(v, [25, 75]))


def main():
    print(f"reading {INDIR}")
    W = load_all()
    W = merge_dropin(W)
    # de-duplicated, order preserved
    have = [c for c in dict.fromkeys(METRICS + KCOLS) if c in W.columns]

    # --- session-level medians: the unit of analysis -------------------------
    sess = (W.groupby(["session", "animal", "state"], observed=True)[have]
              .median().reset_index())
    counts = (W.groupby(["session", "state"], observed=True)
                .size().rename("n_windows").reset_index())
    sess = sess.merge(counts, on=["session", "state"], how="left")

    assert_local_path(OUTDIR)
    os.makedirs(OUTDIR, exist_ok=True)
    sess.to_csv(assert_local_path(os.path.join(OUTDIR, "fig1_session_medians.csv")),
                index=False)

    # --- coverage: how many sessions actually contribute each state ----------
    print("\n=== coverage (sessions contributing, windows pooled) ===")
    cov = (sess.groupby("state")
               .agg(n_sessions=("session", "nunique"),
                    n_windows=("n_windows", "sum"),
                    median_windows_per_session=("n_windows", "median"))
               .reindex([s for s in STATES if s in set(sess.state)]))
    print(cov.to_string())

    # --- group summary: median [IQR] across sessions -------------------------
    rows = []
    for st in [s for s in STATES if s in set(sess.state)]:
        sub = sess[sess.state == st]
        for mcol in have:
            v = sub[mcol].to_numpy(float)
            v = v[np.isfinite(v)]
            rows.append({"state": st, "metric": mcol, "n_sessions": v.size,
                         "median": np.median(v) if v.size else np.nan,
                         "q25": np.percentile(v, 25) if v.size else np.nan,
                         "q75": np.percentile(v, 75) if v.size else np.nan})
    grp = pd.DataFrame(rows)
    grp.to_csv(assert_local_path(os.path.join(OUTDIR, "fig1_group_summary.csv")),
               index=False)

    print("\n=== group medians across sessions [IQR] ===")
    show = [c for c in ["d2", "BR_pyr_k200", "BR_all_k200", "tau_ms_pyr_k200",
                        "MUA_mean", "EI_logratio", "synchrony_raw_mean",
                        "isi_entropy_pooled"] if c in have]
    piv = grp[grp.metric.isin(show)].pivot(index="state", columns="metric",
                                           values="median")
    print(piv.reindex([s for s in STATES if s in piv.index])[show].round(4).to_string())

    # --- k_max sweep, per population ----------------------------------------
    order_states = [s for s in STATES if s in set(sess.state)]
    sweeps = {}
    for pop in POPS:
        cols = [f"BR_{pop}_k{k}" for k in K_MAX_SWEEP if f"BR_{pop}_k{k}" in have]
        if not cols:
            continue
        sweep = sess.groupby("state")[cols].median().reindex(order_states)
        sweeps[pop] = sweep
        print(f"\n=== k_max sweep, {pop} units: median m across sessions ===")
        print(sweep.round(4).to_string())

        tau = -BIN_BR_S * 1000.0 / np.log(sweep.clip(upper=0.999999))
        tau.columns = [c.replace("BR_", "tau_ms_") for c in cols]
        print(f"--- as correlation time tau (ms), {pop} ---")
        print(tau.round(1).to_string())
        sweeps[pop + "_tau"] = tau

        # m is critical at 1; higher m = closer to criticality. Sorted descending
        # so this reads closest -> furthest, matching the d2 ordering.
        print(f"--- states by m, {pop}: CLOSEST -> FURTHEST from criticality "
              f"(descending m) ---")
        for c in cols:
            order = sweep[c].sort_values(ascending=False).index.tolist()
            print(f"    {c:16s}  " + "  <  ".join(order))

    if sweeps:
        pd.concat(sweeps.values(), axis=1).to_csv(
            assert_local_path(os.path.join(OUTDIR, "fig1_kmax_sweep.csv")))

    # --- d2 vs BR: agreement within state ------------------------------------
    # BR_all_k200 (no cell-type/rate filter) is population-matched to d2; BR_pyr_k200
    # is the Fig. 2g population. k200 matches d2's 800 ms kernel; k2500 is W&P's rat
    # fit range. Both are reported.
    br_cols = [c for c in ("BR_all_k200", "BR_pyr_k200",
                           "BR_all_k2500", "BR_pyr_k2500") if c in W.columns]
    if "d2" in W.columns and W["d2"].notna().any() and br_cols:
        rows = []
        for br in br_cols:
            for (s, st), sub in W.groupby(["session", "state"], observed=True):
                v = sub[["d2", br]].dropna()
                if len(v) >= 10:
                    r = np.corrcoef(v["d2"], v[br])[0, 1]
                    rows.append({"session": s, "state": st, "br_col": br,
                                 "pop": "all" if "_all_" in br else "pyr",
                                 "n": len(v), "pearson_r": r})
        cor = pd.DataFrame(rows)
        if len(cor):
            cor.to_csv(assert_local_path(os.path.join(OUTDIR, "fig1_d2_vs_br.csv")),
                       index=False)
            for br in br_cols:
                tag = ("population-MATCHED to d2 -- primary for R2.4"
                       if "_all_" in br else "pyramidal only -- Fig. 2g arm")
                print(f"\n=== within-state d2 ~ {br} correlation   [{tag}] ===")
                print(cor[cor.br_col == br].groupby("state")["pearson_r"]
                         .agg(["count", "median", "min", "max"]).round(3).to_string())
            print("\n  |r| well below 1 => d2 is not a restatement of the branching"
                  "\n  ratio; the two carry partly independent information (R2.4).")
    else:
        print("\n[!] no d2 joined -- skipping the d2-vs-BR comparison")

    # --- synchrony subsample target -----------------------------------------
    if "n_E_units" in W.columns:
        ne = W.groupby("session")["n_E_units"].first()
        print(f"\n=== n_E_units across sessions: min {ne.min()}, median "
              f"{ne.median():.0f}, max {ne.max()} ===")
        print(f"  a fixed subsample target of {int(ne.min())} would be feasible in "
              "every session (open item 5)")

    print(f"\nwrote -> {OUTDIR}")


if __name__ == "__main__":
    main()
