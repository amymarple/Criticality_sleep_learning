"""Drawing functions from analysis_scripts/revision/ (fig1_state_space_umap.py) used by the
figure notebooks.
"""
import matplotlib.pyplot as plt

# matplotlib settings made by importing the scripts
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none",
})




# --- fig1_state_space_umap.py
FEATURES = [
    "MUA_mean", "MUA_fano",
    "EI_logratio",
    "synchrony_sub81_mean",
    "isi_entropy_pooled",
    "psd_slope_20_45", "sw_broadband",
    "theta_ratio",                     # theta power above 1/f
    "ripple_rate", "ripple_amp_mean",
]


# --- fig1_state_space_umap.py
STATE_LABEL = {"THETA": "Wake \u03b8", "nonTHETA": "Wake non-\u03b8",
               "REM": "REM", "NREM": "NREM"}


# --- fig1_state_space_umap.py
STATE_COLOR = {"THETA": "#d95f0e", "nonTHETA": "#fdb863",
               "REM": "#d7b5d8", "NREM": "#7b3294"}
