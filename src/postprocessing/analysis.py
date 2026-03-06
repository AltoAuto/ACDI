"""
postprocessing/analysis.py
--------------------------
ME 5351 HW2 - Quantitative analysis and error metrics.

All error norms are computed in physical space (i.e., integrated over the
domain with the cell area dx * dy as quadrature weight) to give
mesh-independent, dimensionless error measures suitable for convergence studies.

Functions
---------
compute_l1_error
    Integral (L1) norm of |phi - phi_exact|.

compute_l2_error
    Root-mean-square (L2) norm.

compute_linf_error
    Maximum pointwise error.

compute_mass
    Total "mass" (integral of phi over the domain).

compute_interface_width
    Estimate the current diffuse-interface half-thickness by fitting a tanh
    profile to the phi field along a cross-section through the drop centre.

print_error_table
    Pretty-print a Markdown-formatted table of errors for all tasks, useful
    for inclusion in the homework report.
"""

import numpy as np
from typing import Optional

from core.mesh import Mesh


def compute_l1_error(
    phi: np.ndarray,
    phi_exact: np.ndarray,
    mesh: Mesh,
) -> float:
    """Compute the L1 error norm integrated over the domain.

    L1 = sum(|phi - phi_exact|) * dx * dy

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Numerical solution.
    phi_exact : np.ndarray, shape (ny, nx)
        Reference / exact solution.
    mesh : Mesh
        Computational mesh (provides dx, dy).

    Returns
    -------
    l1 : float
    """
    return np.sum(np.abs(phi - phi_exact)) * mesh.dx * mesh.dy


def compute_l2_error(
    phi: np.ndarray,
    phi_exact: np.ndarray,
    mesh: Mesh,
) -> float:
    """Compute the L2 error norm integrated over the domain.

    L2 = sqrt( sum((phi - phi_exact)^2) * dx * dy )

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
    phi_exact : np.ndarray, shape (ny, nx)
    mesh : Mesh

    Returns
    -------
    l2 : float
    """
    return np.sqrt(np.sum((phi - phi_exact)**2) * mesh.dx * mesh.dy)


def compute_linf_error(
    phi: np.ndarray,
    phi_exact: np.ndarray,
) -> float:
    """Compute the Linf (maximum) error.

    Linf = max(|phi - phi_exact|)

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
    phi_exact : np.ndarray, shape (ny, nx)

    Returns
    -------
    linf : float
    """
    return np.max(np.abs(phi - phi_exact))


def compute_mass(
    phi: np.ndarray,
    mesh: Mesh,
) -> float:
    """Compute the total "mass" (volume integral of phi).

    M = sum(phi) * dx * dy

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    mesh : Mesh

    Returns
    -------
    mass : float
    """
    return np.sum(phi) * mesh.dx * mesh.dy


def compute_interface_width(
    phi: np.ndarray,
    mesh: Mesh,
    direction: str = "x",
    j_slice: Optional[int] = None,
) -> float:
    """Estimate the diffuse-interface half-thickness by tanh profile fitting.

    Extracts a 1-D slice through the drop interface along the given direction
    and fits  phi(s) = 0.5 * (1 + tanh((s0 - s) / (2*eps)))  to determine eps.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    mesh : Mesh
        Computational mesh.
    direction : {'x', 'y'}
        Direction of the 1-D slice.
    j_slice : int or None
        Row (if direction='x') or column (if direction='y') index for the
        slice.  If None, uses the row/column passing through the domain centre.

    Returns
    -------
    eps_fit : float
        Fitted interface half-thickness.
    """
    from scipy.optimize import curve_fit

    if direction == "x":
        if j_slice is None:
            j_slice = mesh.ny // 2
        s = mesh.xc
        phi_1d = phi[j_slice, :]
    else:
        if j_slice is None:
            j_slice = mesh.nx // 2
        s = mesh.yc
        phi_1d = phi[:, j_slice]

    # Find approximate interface location (where phi crosses 0.5)
    s0_guess = s[np.argmin(np.abs(phi_1d - 0.5))]

    def tanh_profile(s, s0, eps_fit):
        return 0.5 * (1.0 + np.tanh((s0 - s) / (2.0 * eps_fit)))

    try:
        popt, _ = curve_fit(tanh_profile, s, phi_1d, p0=[s0_guess, mesh.dx * 1.5])
        return abs(popt[1])
    except RuntimeError:
        return float("nan")


def print_error_table(
    results: dict[str, dict[str, float]],
    title: str = "Error Summary",
) -> None:
    """Pretty-print a Markdown error table to stdout.

    Parameters
    ----------
    results : dict
        Outer key: task label (e.g. 'Task 1 - Upwind').
        Inner dict: error metrics, e.g. {'L1': 1e-3, 'L2': 2e-3, 'Linf': 5e-3,
                                          'mass_error': 1e-12}.
    title : str
        Table title printed above the table.
    """
    print(f"\n### {title}\n")

    # Determine columns from first entry
    if not results:
        print("(no results)")
        return

    cols = list(next(iter(results.values())).keys())
    header = "| Method | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)

    print(header)
    print(sep)
    for label, metrics in results.items():
        row = f"| {label} | "
        row += " | ".join(f"{metrics[c]:.4e}" for c in cols)
        row += " |"
        print(row)
    print()
