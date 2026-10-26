"""Drawing functions from analysis_scripts/revision/ (m25_tmaze_box.py) used by the
figure notebooks.
"""
import matplotlib.pyplot as plt

# matplotlib settings made by importing the scripts
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "svg.fonttype": "none",
})
plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none"})




# --- m25_tmaze_box.py
MOUSE_COL = {"PPP4": "#8c564b", "PPP7": "#d62728", "PPP8": "#9467bd", "PPP20": "#1f77b4", "PPP21": "#ff7f0e",
             "PPP23": "#2ca02c"}


# --- m25_tmaze_box.py
MOUSE_LABEL = {"PPP4": "Mouse 1", "PPP7": "Mouse 2", "PPP8": "Mouse 3", "PPP20": "Mouse 4", "PPP21": "Mouse 5",
               "PPP23": "Mouse 6"}
