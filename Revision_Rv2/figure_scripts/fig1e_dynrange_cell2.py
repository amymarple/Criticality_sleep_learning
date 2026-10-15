# Code cell 2 of analysis_scripts/notebooks/figures/Fig1_E_MUA_d2_correlation_FigS1_G_plotting_dynrange.ipynb, executed by FigS2_AF_robustness.ipynb.
import numpy as np
import matplotlib.pyplot as plt

# =======================================
# 1) Prep df_base and MUA column
# =======================================

df_base = all_pairs.copy()

# per-file identifier
if "__file__" not in df_base.columns:
    df_base["__file__"] = df_base["animal"].astype(str) + "_" + df_base["day"].astype(str)

# pick the MUA column name
mua_col = None
for cand in ["MUA", "mua30", "mua", "MUA30"]:
    if cand in df_base.columns:
        mua_col = cand
        break

if mua_col is None:
    raise KeyError(
        "Could not find an MUA column in all_pairs. "
        "Looked for: 'MUA', 'mua30', 'mua', 'MUA30'."
    )

df_base = df_base.rename(columns={mua_col: "MUA"})

# =======================================
# 2) State definitions & colors
# =======================================

state_label_sets = {
    "NREM":      {"NREM"},
    "REM":       {"REM", "REM4"},
    "THETA":     {"THETA", "Wake θ"},
    "non-THETA": {"nonTHETA", "Wake non-θ"},
}

method = "minmax"   # or 'percentile' / 'z'

colors_cond = {
    "NREM":      "#7b3294",
    "REM":       "#d7b5d8",
    "THETA":     "#d95f0e",
    "non-THETA": "#fdb863",
}

# raw summaries (for printing/stats) and per-state normalized summaries (for plotting)
summaries_raw  = {}
summaries_norm = {}

# =======================================
# 3) Per-state summaries + monotonicity
# =======================================
def p_to_stars(p):
    p = float(p)
    if not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"

for label, groups in state_label_sets.items():
    mask = df_base["group"].isin(groups)
    dstate = df_base.loc[mask, ["d2", "MUA", "__file__"]].dropna()

    if dstate.empty:
        print(f"[WARN] {label}: no data, skipping.")
        continue

    # --- dynamic range per file, per d2 bin ---
    df_perfile = perfile_dyn_by_bin(dstate, method=method)
    df_summary = summarize_mean_sem(df_perfile)
    summaries_raw[label] = df_summary


    # --- per-state 0–1 normalization just for plotting ---
    if not df_summary.empty:
        dfN = df_summary.copy()
        m = dfN["mean"].to_numpy()
        mn = np.nanmin(m)
        mx = np.nanmax(m)

        if np.isfinite(mn) and np.isfinite(mx) and not np.isclose(mn, mx):
            dfN["mean"] = (dfN["mean"] - mn) / (mx - mn)
            dfN["sem"]  = dfN["sem"] / (mx - mn)
        else:
            # degenerate case: all values equal → set mean=0.5, sem=0
            dfN["mean"] = 0.5
            dfN["sem"]  = 0.0

        summaries_norm[label] = dfN
    else:
        summaries_norm[label] = df_summary

    # --- monotonicity test via Spearman + Stouffer meta (use RAW, not normalized) ---
    T = perfile_dyn_by_bin_no_baseline(dstate, method=method)
    res = spearman_stouffer(T, alternative="decreasing")  # dyn decreasing with d2?
    if res is not None:
        print(f"  {label}: per-session Spearman rho combined by Stouffer's method (one-sided), "
              f"rho = {res['rho_wmean_fisher']:+.2f}, p = {res['p_one_sided']:.2g}")
    else:
        print(f"{label}: not enough bins/files for meta-analysis.")

# =======================================
# 4) Combined plots using per-state normalized dynamic range
# =======================================

# --- (A) NREM vs REM ---

plt.figure(figsize=(4, 6))
ax = plt.gca()

for label in ["NREM", "REM"]:
    if label not in summaries_norm:
        continue
    dfS = summaries_norm[label]
    if dfS.empty:
        continue

    # x-axis: per-state normalized (or z-scored) mean dynamic range
    x = dfS["mean"].to_numpy(float)
    e = dfS["sem"].to_numpy(float)

    # y-axis: bin index 1..6
    if "d2_bin" in dfS.columns:
        y = dfS["d2_bin"].to_numpy(int) + 1   # if 0-based
    elif "tile_num" in dfS.columns:
        y = dfS["tile_num"].to_numpy(int)     # if already 1..6
    else:
        raise KeyError("Need a bin column like 'd2_bin' or 'tile_num' in dfS to plot y=1..6.")

    ax.errorbar(
        x, y,
        xerr=e,
        fmt='-o',
        capsize=4,
        linewidth=2,
        label=label,
        color=colors_cond.get(label, "k"),
    )

# y-axis formatting for 6 bins
ax.set_ylim(0.5, 6.5)

ax.set_xlabel("normalized MUA dynamic range")
ax.set_ylabel("d² bin")
ax.set_title("Dynamic range vs d² (NREM / REM)")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)
plt.tight_layout()

save_path = rf"D:\Documents\GitHub\Sleep_criticaltiy\reports\new_planc\individual_fig_main\fig1_E_NREM_vs_REM_dyn_by_d2_{method}_norm.svg"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()


# --- (B) THETA vs non-THETA ---
plt.figure(figsize=(4, 6))
ax = plt.gca()

for label in ["THETA", "non-THETA"]:
    if label not in summaries_norm:
        continue
    dfS = summaries_norm[label]
    if dfS.empty:
        continue

    # x-axis: per-state normalized (or z-scored) mean dynamic range
    x = dfS["mean"].to_numpy(float)
    e = dfS["sem"].to_numpy(float)

    # y-axis: bin index 1..6
    if "d2_bin" in dfS.columns:
        y = dfS["d2_bin"].to_numpy(int) + 1   # if 0-based
    elif "tile_num" in dfS.columns:
        y = dfS["tile_num"].to_numpy(int)     # if already 1..6
    else:
        raise KeyError("Need a bin column like 'd2_bin' or 'tile_num' in dfS to plot y=1..6.")

    ax.errorbar(
        x, y,
        xerr=e,
        fmt='-o',
        capsize=4,
        linewidth=2,
        label=label,
        color=colors_cond.get(label, "k"),
    )

ax.set_xlabel("normalized MUA dynamic range ")
ax.set_ylabel("normalized d² distance")
ax.set_title("Dynamic range vs d² (THETA / non-THETA)")
ax.legend(frameon=False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)
plt.tight_layout()
save_path = rf"D:\Documents\GitHub\Sleep_criticaltiy\reports\new_planc\individual_fig_main\fig1_E_THETA_vs_nonTHETA_dyn_by_d2_{method}_norm.svg"
plt.savefig(save_path, dpi=300, bbox_inches="tight")
plt.show()
