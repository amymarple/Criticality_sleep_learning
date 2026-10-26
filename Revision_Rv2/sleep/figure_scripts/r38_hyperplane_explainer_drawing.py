"""Drawing functions from analysis_scripts/revision/ (d2_yw_matlab_match.py, r38_hyperplane_explainer.py) used by the
figure notebooks. LOG: the folder of the notebook's tables, set by the notebook.
"""
import matplotlib.pyplot as plt
import numpy as np

# matplotlib settings made by importing the scripts
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "svg.fonttype": "none"})

LOG = None


# --- r38_hyperplane_explainer.py
EX = "PPP20_day5"


# --- r38_hyperplane_explainer.py
WIN = 30.0


# --- r38_hyperplane_explainer.py
RED = "#d62728"


# --- r38_hyperplane_explainer.py
COL = {"NREM": "#7b3294", "REM": "#d7b5d8", "THETA": "#d95f0e", "nonTHETA": "#fdb863"}


# --- d2_yw_matlab_match.py
DT, P = 0.04, 20


# --- r38_hyperplane_explainer.py
def model_psd(phi, s2, f, dt=DT):
    """One-sided PSD per Hz of the AR model: 2 s2 dt / |1 - sum phi_k e^{-i 2 pi f k dt}|^2."""
    k = np.arange(1, phi.size + 1)
    h = 1 - np.exp(-2j * np.pi * np.outer(f, k) * dt) @ phi
    return 2 * s2 * dt / np.abs(h) ** 2
