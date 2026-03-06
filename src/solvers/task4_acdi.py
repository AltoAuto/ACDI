"""
solvers/task4_acdi.py
---------------------
ME 5351 HW2 - Task 4: ACDI (Accurate Conservative Diffuse Interface).

Numerical method
~~~~~~~~~~~~~~~~
  Spatial scheme  : 2nd-order skew-symmetric-like splitting for advection
                    + ACDI regularisation (Jain 2022, Eq. 20-21)
  Time integration: Classical 4th-order Runge-Kutta (RK4)

Governing equation (split form, Jain 2022 Eq. 20)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The ACDI method writes the transport equation as a two-operator split:

    Operator A (skew-symmetric advection):
        d(phi)/dt = -0.5 * [div(u*phi) + u . grad(phi)]
                  = -0.5 * div(u*phi) - 0.5 * u . grad(phi)

    Operator B (ACDI regularisation, Eq. 21):
        d(phi)/dt = div[ eps * phi*(1-phi)*n_hat - 0.5*eps*grad(phi) ]

The skew-symmetric splitting of the advection operator exactly cancels the
spurious divergence error for incompressible (div(u)=0) flows, recovering
conservation while avoiding the oscillations of a pure central scheme.

Physical motivation
~~~~~~~~~~~~~~~~~~~
For div(u) = 0, div(u*phi) = u . grad(phi), so the skew-symmetric form:
    -0.5 * [div(u*phi) + u . grad(phi)]
is mathematically equivalent to the standard form but discretely more
accurate because the anti-symmetric part of the error cancels.

Usage
-----
    from solvers.task4_acdi import run_task4
    phi_history, t_history = run_task4(cfg)

Expected config keys
--------------------
  mesh      : Mesh
  phi0      : np.ndarray, shape (ny, nx)
  velocity  : str  -- 'uniform' | 'shear'
  eps       : float  -- interface half-thickness
  t_end     : float
  dt        : float
  save_freq : int
  output_dir: str  -- path to results/task4/

Reference
---------
  Jain, S. S. (2022). J. Comput. Phys., 469, 111529.  Eq. (20)-(21).
"""

import numpy as np
from typing import Any

from core.mesh import Mesh
from core.flux_schemes import central_flux
from core.regularization import acdi_regularization
from core.time_integration import rk4_step, compute_cfl


def skew_symmetric_advection(
    phi: np.ndarray,
    u_face: np.ndarray,
    v_face: np.ndarray,
    mesh: Mesh,
) -> np.ndarray:
    """Compute the skew-symmetric advection operator (Jain 2022, Eq. 20).

    Returns:
        -0.5 * div(u*phi) - 0.5 * u . grad(phi)

    For divergence-free u, this equals -div(u*phi) exactly in continuous
    form, but the discrete split reduces commutation errors and improves
    the energy-stability properties of the scheme.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    u_face : np.ndarray, shape (ny, nx+1)
        x-velocity at east/west faces.
    v_face : np.ndarray, shape (ny+1, nx)
        y-velocity at north/south faces.
    mesh : Mesh
        Computational mesh.

    Returns
    -------
    adv : np.ndarray, shape (ny, nx)
        Skew-symmetric advection contribution to d(phi)/dt.
    """
    # Term 1: -0.5 * div(u*phi)  using central flux
    div_term = central_flux(phi, u_face, v_face, mesh)  # already returns -div(u*phi)

    # Term 2: -0.5 * u . grad(phi)  at cell centres
    # Interpolate face velocities to cell centres
    # u_cc[j,i] = 0.5*(u_face[j,i] + u_face[j,i+1])
    u_cc = 0.5 * (u_face[:, :-1] + u_face[:, 1:])   # (ny, nx)
    v_cc = 0.5 * (v_face[:-1, :] + v_face[1:, :])    # (ny, nx)

    # Central-difference gradient of phi
    dphi_dx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * mesh.dx)
    dphi_dy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * mesh.dy)

    u_grad_phi = u_cc * dphi_dx + v_cc * dphi_dy

    return 0.5 * div_term - 0.5 * u_grad_phi


def build_rhs(
    phi: np.ndarray,
    t: float,
    mesh: Mesh,
    velocity_fn,
    eps: float,
) -> np.ndarray:
    """Assemble the full ACDI RHS: skew-symmetric advection + ACDI regularisation.

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

    Returns
    -------
    rhs : np.ndarray, shape (ny, nx)
        d(phi)/dt = skew_symmetric_advection + R_ACDI(phi).
    """
    u_face, v_face = velocity_fn(mesh, t)
    adv = skew_symmetric_advection(phi, u_face, v_face, mesh)
    # Gamma >= |u|_max for ACDI
    u_max = max(np.max(np.abs(u_face)), np.max(np.abs(v_face)), 1e-14)
    Gamma = u_max
    reg = acdi_regularization(phi, mesh, eps, Gamma=Gamma)
    return adv + reg


def run_task4(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[float]]:
    """Run the Task 4 solver (ACDI + skew-symmetric + RK4).

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

    def rhs_fn(phi, t):
        return build_rhs(phi, t, mesh, velocity_fn, eps)

    phi_history = [phi.copy()]
    t_history = [0.0]

    t = 0.0
    step = 0
    while t < t_end - 1e-14:
        dt_actual = min(dt, t_end - t)
        phi = rk4_step(phi, t, dt_actual, rhs_fn)
        t += dt_actual
        step += 1

        if step % save_freq == 0 or t >= t_end - 1e-14:
            phi_history.append(phi.copy())
            t_history.append(t)

    return phi_history, t_history
