"""
Criticality measures: neuronal avalanches (av_analysis) and the MR branching-ratio estimator of Wilting and
Priesemann (2018), fetched from the authors' repository by get_wiltingpriesemann.py.
"""

from .av_analysis import (
    coarse_grain_avalanche_detection_all_units_intervals,
    plot_power_law_ccdf,
    plot_dcc,
    compute_average_size,
    compute_scaling_with_avg,
    plot_power_law_pdf
)

__version__ = "0.1.0"
__all__ = [
    "coarse_grain_avalanche_detection_all_units_intervals",
    "plot_power_law_ccdf",
    "plot_dcc",
    "compute_average_size",
    "compute_scaling_with_avg",
    "plot_power_law_pdf",
]

try:
    from . import WiltingPriesemann
    __all__.append("WiltingPriesemann")
except ImportError:
    pass          # python methods/criticality/get_wiltingpriesemann.py
