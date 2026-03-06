"""
solvers/task1_upwind.py
-----------------------
ME 5351 HW2 - Task 1: Plain scalar advection.

Numerical method
~~~~~~~~~~~~~~~~
  Spatial scheme  : 1st-order upwind (donor-cell)
  Time integration: 1st-order explicit Euler

Governing equation (no regularisation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    d(phi)/dt + div(u * phi) = 0

The upwind scheme approximates the face value as the value in the
upstream cell, introducing O(dx) numerical diffusion.  The Euler
scheme advances the field in time using only the current time level.

Stability constraint
~~~~~~~~~~~~~~~~~~~~
    CFL = |u| * dt / dx <= 1  (must be enforced by the caller)

Usage
-----
    from solvers.task1_upwind import run_task1
    phi_history, t_history = run_task1(cfg)

Expected config keys
--------------------
  mesh      : Mesh
  phi0      : np.ndarray, shape (ny, nx)  -- initial condition
  velocity  : str  -- 'uniform' | 'shear'
  t_end     : float
  dt        : float
  save_freq : int  -- save every N steps
  output_dir: str  -- path to results/task1/
"""

import numpy as np
from typing import Any

from core.mesh import Mesh
from core.flux_schemes import upwind_flux
from core.time_integration import euler_step, compute_cfl


def build_rhs(
    phi: np.ndarray,
    t: float,
    mesh: Mesh,
    velocity_fn,
) -> np.ndarray:
    """Assemble the RHS for Task 1: pure upwind advection, no regularisation.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Current volume-fraction field.
    t : float
        Current time (needed for time-varying velocity fields).
    mesh : Mesh
        Computational mesh.
    velocity_fn : callable
        Returns (u_face, v_face) given (mesh, t).

    Returns
    -------
    rhs : np.ndarray, shape (ny, nx)
        d(phi)/dt = -div(u * phi)  evaluated with 1st-order upwind.
    """
    u_face, v_face = velocity_fn(mesh, t)
    return upwind_flux(phi, u_face, v_face, mesh)


def run_task1(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[float]]:
    """Run the Task 1 solver (upwind + Euler) and return the solution history.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.  See module docstring for required keys.

    Returns
    -------
    phi_history : list of np.ndarray
        Snapshots of the volume-fraction field at saved time levels.
    t_history : list of float
        Corresponding simulation times.
    """
    mesh = cfg["mesh"]
    phi = cfg["phi0"].copy()
    velocity_fn = cfg["velocity_fn"]
    t_end = cfg["t_end"]
    dt = cfg["dt"]
    save_freq = cfg.get("save_freq", 10)

    def rhs_fn(phi, t):
        return build_rhs(phi, t, mesh, velocity_fn)

    phi_history = [phi.copy()]
    t_history = [0.0]

    t = 0.0
    step = 0
    while t < t_end - 1e-14:
        dt_actual = min(dt, t_end - t)
        phi = euler_step(phi, t, dt_actual, rhs_fn)
        t += dt_actual
        step += 1

        if step % save_freq == 0 or t >= t_end - 1e-14:
            phi_history.append(phi.copy())
            t_history.append(t)

    return phi_history, t_history
