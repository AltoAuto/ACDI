"""
test_cases/shear_flow.py
------------------------
ME 5351 HW2 - Test Case 2: Circular drop in oscillating shear flow.

Problem setup
~~~~~~~~~~~~~
  Domain       : [0, 1]^2 (unit square, periodic BCs)
  Initial cond.: Circular drop centred at (0.5, 0.75), radius R = 0.15
  Velocity     : Oscillating shear (LeVeque 1996 / Rider & Kothe 1998):
                     u =  sin^2(pi*x) * sin(2*pi*y) * cos(pi*t / T)
                     v = -sin(2*pi*x) * sin^2(pi*y) * cos(pi*t / T)
  Period       : T = 2.0
  End time     : t_end = T = 2.0  (drop should return to initial position)

This is a stringent test because the shear flow stretches the drop into a
thin filament that wraps around the domain.  At t = T the flow reverses and
the drop should recover its initial circular shape.  Numerical errors
manifest as:
  - Residual deformation at t = T (shape error)
  - Loss of mass (conservation error)
  - Spurious filament breakup (topology error)

Expected behaviour by task
~~~~~~~~~~~~~~~~~~~~~~~~~~
  Task 1 : Severe smearing; drop nearly disappears as filament diffuses away.
  Task 2 : CDI prevents mass loss but 1st-order smearing is still significant.
  Task 3 : 2nd-order accuracy improves recovery; thin filament better resolved.
  Task 4 : ACDI best preserves the drop shape at t = T with minimal error.

Error metrics computed (at t = T)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  - L1, L2, Linf errors vs. initial condition (exact solution at t = T)
  - Mass error: |integral(phi_T) - integral(phi0)|
  - Interface area error (optional): |perimeter(phi_T) - perimeter(phi0)|

Reference
---------
  LeVeque, R. J. (1996). J. Comput. Phys., 123, 187-192.
  Rider, W. J. & Kothe, D. B. (1998). J. Comput. Phys., 141, 112-152.
  Jain 2022, Section 4.2.
"""

import numpy as np
from typing import Any

from core.mesh import Mesh
from core.initial_conditions import circular_drop
from core.velocity_fields import shear_flow_velocity


def setup(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the mesh, initial condition, and velocity for the shear flow test.

    Parameters
    ----------
    cfg : dict
        Must contain: nx, ny, Lx, Ly, R, eps, T (period).

    Returns
    -------
    setup_dict : dict
        Keys: 'mesh', 'phi0', 'velocity_fn'.
    """
    mesh = Mesh(
        nx=cfg["nx"], ny=cfg["ny"],
        Lx=cfg["Lx"], Ly=cfg["Ly"],
        x0=cfg.get("x0", 0.0), y0=cfg.get("y0", 0.0),
    )
    x_drop = cfg.get("x_drop", 0.5)
    y_drop = cfg.get("y_drop", 0.75)
    R = cfg["R"]
    eps = cfg["eps"]
    T = cfg.get("T_period", 2.0)

    phi0 = circular_drop(mesh, x_drop, y_drop, R, eps)

    def velocity_fn(mesh, t):
        return shear_flow_velocity(mesh, t, T=T)

    return {
        "mesh": mesh,
        "phi0": phi0,
        "velocity_fn": velocity_fn,
    }


def compute_errors(
    phi: np.ndarray,
    phi0: np.ndarray,
    mesh: Mesh,
) -> dict[str, float]:
    """Compute errors at t = T relative to the initial condition.

    Since the exact solution at t = T is phi0 (periodic flow reversal),
    this measures the reversibility error of each method.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Numerical solution at t = T.
    phi0 : np.ndarray, shape (ny, nx)
        Initial condition (= exact solution at t = T).
    mesh : Mesh
        Computational mesh.

    Returns
    -------
    errors : dict
        Keys: 'L1', 'L2', 'Linf', 'mass_error'.
    """
    diff = np.abs(phi - phi0)
    dA = mesh.dx * mesh.dy
    L1 = np.sum(diff) * dA
    L2 = np.sqrt(np.sum(diff**2) * dA)
    Linf = np.max(diff)
    mass_num = np.sum(phi) * dA
    mass_0 = np.sum(phi0) * dA
    mass_error = abs(mass_num - mass_0)

    return {"L1": L1, "L2": L2, "Linf": Linf, "mass_error": mass_error}


def run_shear_flow(
    task_id: int,
    cfg: dict[str, Any],
) -> tuple[list[np.ndarray], list[float], dict[str, float]]:
    """Run the oscillating shear flow test for the specified task solver.

    Parameters
    ----------
    task_id : int
        Which solver to use (1, 2, 3, or 4).
    cfg : dict
        Full configuration dictionary (merged with test-case defaults).

    Returns
    -------
    phi_history : list of np.ndarray
        Time snapshots of phi.
    t_history : list of float
        Corresponding simulation times.
    metrics : dict
        Error metrics at t = T.
    """
    from solvers import run_task1, run_task2, run_task3, run_task4

    setup_dict = setup(cfg)
    cfg["mesh"] = setup_dict["mesh"]
    cfg["phi0"] = setup_dict["phi0"]
    cfg["velocity_fn"] = setup_dict["velocity_fn"]

    solver_map = {1: run_task1, 2: run_task2, 3: run_task3, 4: run_task4}
    solver = solver_map[task_id]

    phi_history, t_history = solver(cfg)

    # At t = T, exact solution = initial condition
    metrics = compute_errors(phi_history[-1], setup_dict["phi0"], setup_dict["mesh"])

    return phi_history, t_history, metrics
