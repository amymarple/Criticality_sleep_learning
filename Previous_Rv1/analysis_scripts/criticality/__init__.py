"""
Sleep criticality analysis package.
"""

from .av_analysis import (
    coarse_grain_avalanche_detection_all_units_intervals,
    plot_power_law_ccdf,
    plot_dcc,
    compute_average_size,
    compute_scaling_with_avg,
    plot_power_law_pdf
)

from . import WiltingPriesemann

__version__ = "0.1.0"
__all__ = [
    "coarse_grain_avalanche_detection_all_units_intervals",
    "plot_power_law_ccdf",
    "plot_dcc",
    "compute_average_size",
    "compute_scaling_with_avg",
    "plot_power_law_pdf",
    "WiltingPriesemann"
]
