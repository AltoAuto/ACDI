"""
solvers/task2_cdi_1st.py
------------------------
ME 5351 HW2 - Task 2: CDI with 1st-order spatial and temporal discretisation.

Numerical method
~~~~~~~~~~~~~~~~
  Spatial scheme  : 1st-order upwind (donor-cell) for advection
  Regularisation  : CDI (Conservative Diffuse Interface), Jain 2022 Eq. (6)-(8)
  Time integration: 1st-order explicit Euler

Governing equation
~~~~~~~~~~~~~~~~~~
    d(phi)/dt + div(u * phi) = R_CDI(phi)

where the CDI regularisation term is:
    R_CDI = div[ eps * phi*(1-phi)*n_hat ] - eps * lap(phi)

This keeps the interface width locked to eps while remaining globally
conservative (no net change in the integral of phi from the regularisation).

Stability constraint
~~~~~~~~~~~~~~~~~~~~
    CFL_adv = |u| * dt / dx  <= 1
    CFL_reg = eps * dt / dx^2 <= 0.5  (diffusive stability for the lap term)

Both constraints must be checked; typically the regularisation is the
more restrictive one for fine meshes.

Usage
-----
    from solvers.task2_cdi_1st import run_task2
    phi_history, t_history = run_task2(cfg)

Expected config keys
--------------------
  mesh      : Mesh
  phi0      : np.ndarray, shape (ny, nx)
  velocity  : str  -- 'uniform' | 'shear'
  eps       : float  -- interface half-thickness
  t_end     : float
  dt        : float
  save_freq : int
  output_dir: str  -- path to results/task2/
"""

import numpy as np
from typing import Any

from core.mesh import Mesh
from core.flux_schemes import upwind_flux
from core.regularization import cdi_regularization
from core.time_integration import euler_step, compute_cfl


def build_rhs(
    phi: np.ndarray,
    t: float,
    mesh: Mesh,
    velocity_fn,
    eps: float,
    u_max_global: float,
) -> np.ndarray:
    """Assemble the CDI RHS: upwind advection + CDI regularisation.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Current volume-fraction field.
    t : float
        Current time.
    mesh : Mesh
        Computational mesh.
    velocity_fn : callable
        Returns (u_face, v_face) given (mesh, t).
    eps : float
        Interface half-thickness parameter.
    u_max_global : float
        Global maximum velocity magnitude (over all time).

    Returns
    -------
    rhs : np.ndarray, shape (ny, nx)
        d(phi)/dt = -div(u*phi) + R_CDI(phi).
    """
    u_face, v_face = velocity_fn(mesh, t)
    adv = upwind_flux(phi, u_face, v_face, mesh)
    # Gamma >= |u|_max / (2*eps/dx - 1), using global max velocity
    eps_star = eps / mesh.dx
    Gamma = u_max_global / max(2.0 * eps_star - 1.0, 0.1)
    reg = cdi_regularization(phi, mesh, eps, Gamma=Gamma)
    return adv + reg


def run_task2(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[float]]:
    """Run the Task 2 solver (CDI + upwind + Euler) and return solution history.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary.  See module docstring for required keys.

    Returns
    -------
    phi_history : list of np.ndarray
        Snapshots of the volume-fraction field.
    t_history : list of float
        Corresponding simulation times.
    """
    mesh = cfg["mesh"]
    phi = cfg["phi0"].copy()
    velocity_fn = cfg["velocity_fn"]
    eps = cfg["eps"]
    t_end = cfg["t_end"]
    dt = cfg["dt"]
    save_freq = cfg.get("save_freq", 10)

    # Global max velocity (at t=0 when cos(pi*t/T)=1 for shear flow)
    u_face0, v_face0 = velocity_fn(mesh, 0.0)
    u_max_g = max(np.max(np.abs(u_face0)), np.max(np.abs(v_face0)), 1e-14)

    # Euler stability limit for CDI regularisation.
    # The maximum eigenvalue of the CDI operator is approximately
    # lambda_max ~ -8*Gamma*eps/dx^2 (sharpening term doubles the pure-diffusion
    # stiffness).  Euler requires dt < 1/|lambda_max|; we use 90% for safety.
    eps_star = eps / mesh.dx
    Gamma = u_max_g / max(2.0 * eps_star - 1.0, 0.1)
    dt_diff_max = 0.9 * mesh.dx ** 2 / (8.0 * Gamma * eps)
    if dt > dt_diff_max:
        dt = dt_diff_max

    def rhs_fn(phi, t):
        return build_rhs(phi, t, mesh, velocity_fn, eps, u_max_g)

    phi_history = [phi.copy()]
    t_history = [0.0]

    t = 0.0
    step = 0
    while t < t_end - 1e-14:
        dt_actual = min(dt, t_end - t)
        phi = np.clip(euler_step(phi, t, dt_actual, rhs_fn), 0.0, 1.0)
        t += dt_actual
        step += 1

        if step % save_freq == 0 or t >= t_end - 1e-14:
            phi_history.append(phi.copy())
            t_history.append(t)

    return phi_history, t_history
