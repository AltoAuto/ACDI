"""
postprocessing/__init__.py
--------------------------
ME 5351 HW2 - Post-processing utilities.

Provides:
  - plotting  : Visualisation of phi fields, interface contours, convergence plots.
  - analysis  : Quantitative metrics (L1/L2/Linf errors, mass, interface width).
"""

from .plotting import (
    plot_phi_field,
    plot_interface_contour,
    plot_convergence,
    plot_mass_history,
    animate_phi,
)
from .analysis import (
    compute_l1_error,
    compute_l2_error,
    compute_linf_error,
    compute_mass,
    compute_interface_width,
    print_error_table,
)
