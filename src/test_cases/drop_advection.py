"""
test_cases/drop_advection.py
----------------------------
ME 5351 HW2 - Test Case 1: Circular drop in uniform advection flow.

Problem setup
~~~~~~~~~~~~~
  Domain       : [0, 1]^2 (unit square, periodic BCs)
  Initial cond.: Circular drop centred at (0.5, 0.5), radius R = 0.15
  Velocity     : u = 5, v = 0 (uniform, constant)
  End time     : t_end = 0.1  (drop advances 0.5 domain lengths)
  Exact soln   : Shifted initial condition -- phi_exact(x, y, t) = phi0(x - U0*t, y)

The test measures how well each solver preserves the drop shape under
advection.  Expected behaviour by task:

  Task 1 (upwind + Euler)       : Strong numerical diffusion, interface smears.
  Task 2 (CDI + upwind + Euler) : CDI sharpens the interface; less smearing.
  Task 3 (CDI + central + RK4)  : 2nd-order accuracy, sharper profile.
  Task 4 (ACDI + skew + RK4)    : Best accuracy; minimal shape distortion.

Error metrics computed
~~~~~~~~~~~~~~~~~~~~~~
  - L1 error  : sum(|phi - phi_exact|) * dx * dy
  - L2 error  : sqrt(sum((phi - phi_exact)^2) * dx * dy)
  - Linf error: max(|phi - phi_exact|)
  - Mass error: |integral(phi) - integral(phi0)| (conservation check)
"""

import numpy as np
from typing import Any

from core.mesh import Mesh
from core.initial_conditions import circular_drop
from core.velocity_fields import uniform_velocity


def setup(cfg: dict[str, Any]) -> dict[str, Any]:
    """Build the mesh, initial condition, and velocity for the drop advection test.

    Parameters
    ----------
    cfg : dict
        Must contain: nx, ny, Lx, Ly, R, eps, U0.

    Returns
    -------
    setup_dict : dict
        Keys: 'mesh', 'phi0', 'velocity_fn', 'phi_exact_fn'.
    """
    mesh = Mesh(
        nx=cfg["nx"], ny=cfg["ny"],
        Lx=cfg["Lx"], Ly=cfg["Ly"],
        x0=cfg.get("x0", 0.0), y0=cfg.get("y0", 0.0),
    )
    x_drop = cfg.get("x_drop", 0.5)
    y_drop = cfg.get("y_drop", 0.5)
    R = cfg["R"]
    eps = cfg["eps"]
    U0 = cfg.get("U0", 5.0)
    V0 = cfg.get("V0", 0.0)

    phi0 = circular_drop(mesh, x_drop, y_drop, R, eps)

    def velocity_fn(mesh, t):
        return uniform_velocity(mesh, U0=U0, V0=V0)

    def phi_exact_fn(mesh, t):
        return phi_exact(mesh, t, x_drop, y_drop, R, eps, U0)

    return {
        "mesh": mesh,
        "phi0": phi0,
        "velocity_fn": velocity_fn,
        "phi_exact_fn": phi_exact_fn,
    }


def phi_exact(
    mesh: Mesh,
    t: float,
    x0: float,
    y0: float,
    R: float,
    eps: float,
    U0: float,
) -> np.ndarray:
    """Return the exact solution at time t (shifted initial condition).

    The exact solution is the initial drop profile translated by U0 * t in x,
    with periodic wrapping applied so the drop re-enters from the left edge.

    Parameters
    ----------
    mesh : Mesh
        Computational mesh.
    t : float
        Current simulation time.
    x0, y0 : float
        Initial drop centre.
    R : float
        Drop radius.
    eps : float
        Interface half-thickness.
    U0 : float
        Advection velocity.

    Returns
    -------
    phi_ex : np.ndarray, shape (ny, nx)
        Exact volume-fraction field at time t.
    """
    # Shifted centre with periodic wrap
    x_new = x0 + U0 * t
    # Wrap into domain [mesh.x0, mesh.x0 + mesh.Lx)
    x_new = mesh.x0 + (x_new - mesh.x0) % mesh.Lx
    return circular_drop(mesh, x_new, y0, R, eps)


def compute_errors(
    phi: np.ndarray,
    phi_ex: np.ndarray,
    mesh: Mesh,
) -> dict[str, float]:
    """Compute L1, L2, Linf, and mass conservation errors.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Numerical solution.
    phi_ex : np.ndarray, shape (ny, nx)
        Exact solution.
    mesh : Mesh
        Computational mesh (provides dx, dy for integration weights).

    Returns
    -------
    errors : dict
        Keys: 'L1', 'L2', 'Linf', 'mass_error'.
    """
    diff = np.abs(phi - phi_ex)
    dA = mesh.dx * mesh.dy
    L1 = np.sum(diff) * dA
    L2 = np.sqrt(np.sum(diff**2) * dA)
    Linf = np.max(diff)
    mass_num = np.sum(phi) * dA
    mass_ex = np.sum(phi_ex) * dA
    mass_error = abs(mass_num - mass_ex)

    return {"L1": L1, "L2": L2, "Linf": Linf, "mass_error": mass_error}


def run_drop_advection(
    task_id: int,
    cfg: dict[str, Any],
) -> tuple[list[np.ndarray], list[float], dict[str, float]]:
    """Run the drop advection test for the specified task solver.

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
        Error metrics at final time.
    """
    from solvers import run_task1, run_task2, run_task3, run_task4

    # Setup mesh, IC, velocity
    setup_dict = setup(cfg)
    cfg["mesh"] = setup_dict["mesh"]
    cfg["phi0"] = setup_dict["phi0"]
    cfg["velocity_fn"] = setup_dict["velocity_fn"]

    solver_map = {1: run_task1, 2: run_task2, 3: run_task3, 4: run_task4}
    solver = solver_map[task_id]

    phi_history, t_history = solver(cfg)

    # Compute errors at final time
    phi_ex = setup_dict["phi_exact_fn"](setup_dict["mesh"], t_history[-1])
    metrics = compute_errors(phi_history[-1], phi_ex, setup_dict["mesh"])

    return phi_history, t_history, metrics
