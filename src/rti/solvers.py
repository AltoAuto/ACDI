"""
rti/solvers.py
--------------
RTI time-stepping loops using CDI (Task 3 approach) and ACDI (Task 4 approach).

Both solvers:
  - Compute the Stokes velocity from the current φ field each RHS evaluation
  - Use adaptive Gamma that tracks the instantaneous velocity magnitude
  - Advance with classical RK4
  - Return (phi_history, t_history) for post-processing
"""

import numpy as np
from typing import Any

from core.mesh import Mesh
from core.flux_schemes import central_flux
from core.regularization import cdi_regularization, acdi_regularization
from core.time_integration import rk4_step
from solvers.task4_acdi import skew_symmetric_advection
from rti.velocity import stokes_velocity


# ---------------------------------------------------------------------------
# RHS builders
# ---------------------------------------------------------------------------

def build_rhs_cdi(
    phi: np.ndarray,
    mesh: Mesh,
    eps: float,
    g_eff: float,
) -> np.ndarray:
    """CDI RHS: central advection + CDI regularisation with adaptive Gamma.

    Gamma criterion (Jain 2022, Eq. 9 for CDI):
        Gamma >= |u|_max / (2*eps/dx - 1)
    """
    u_face, v_face = stokes_velocity(phi, mesh, g_eff)
    u_max = max(float(np.abs(u_face).max()), float(np.abs(v_face).max()), eps * 0.05)
    eps_star = eps / mesh.dx
    Gamma = u_max / max(2.0 * eps_star - 1.0, 0.1)
    adv = central_flux(phi, u_face, v_face, mesh)
    reg = cdi_regularization(phi, mesh, eps, Gamma=Gamma)
    return adv + reg


def build_rhs_acdi(
    phi: np.ndarray,
    mesh: Mesh,
    eps: float,
    g_eff: float,
) -> np.ndarray:
    """ACDI RHS: skew-symmetric advection + ACDI regularisation with adaptive Gamma.

    Gamma criterion (Jain 2022, Eq. 9 for ACDI):
        Gamma >= |u|_max
    """
    u_face, v_face = stokes_velocity(phi, mesh, g_eff)
    u_max = max(float(np.abs(u_face).max()), float(np.abs(v_face).max()), eps * 0.05)
    Gamma = u_max
    adv = skew_symmetric_advection(phi, u_face, v_face, mesh)
    reg = acdi_regularization(phi, mesh, eps, Gamma=Gamma)
    return adv + reg


# ---------------------------------------------------------------------------
# Time-stepping loops
# ---------------------------------------------------------------------------

def _run_rti(cfg: dict[str, Any], rhs_builder, label: str):
    """Generic RTI time-stepping loop (used by both CDI and ACDI).

    Parameters
    ----------
    cfg : dict
        Configuration dict with keys: mesh, phi0, eps, g_eff, t_end, dt, save_freq.
    rhs_builder : callable
        Function with signature (phi, mesh, eps, g_eff) -> rhs.
    label : str
        Solver label for progress output.

    Returns
    -------
    phi_history : list of np.ndarray
    t_history : list of float
    """
    mesh: Mesh = cfg["mesh"]
    phi = cfg["phi0"].copy()
    eps: float = cfg["eps"]
    g_eff: float = cfg["g_eff"]
    t_end: float = cfg["t_end"]
    dt: float = cfg["dt"]
    save_freq: int = cfg.get("save_freq", 20)

    mass0 = float(phi.sum()) * mesh.dx * mesh.dy

    def rhs_fn(phi, t):
        return rhs_builder(phi, mesh, eps, g_eff)

    phi_history = [phi.copy()]
    t_history = [0.0]

    t = 0.0
    step = 0
    total_steps = int(np.ceil(t_end / dt))
    report_every = max(total_steps // 10, 1)

    print(f"[{label}] Running {total_steps} steps to t={t_end}  "
          f"(nx={mesh.nx}, eps={eps:.4f}, g_eff={g_eff})")

    while t < t_end - 1e-14:
        dt_actual = min(dt, t_end - t)
        phi = rk4_step(phi, t, dt_actual, rhs_fn)
        t += dt_actual
        step += 1

        if step % save_freq == 0 or t >= t_end - 1e-14:
            phi_history.append(phi.copy())
            t_history.append(t)

        if step % report_every == 0:
            mass = float(phi.sum()) * mesh.dx * mesh.dy
            mass_err = abs(mass - mass0)
            print(f"  step {step:5d}  t={t:.4f}  "
                  f"phi=[{phi.min():.3f},{phi.max():.3f}]  "
                  f"mass_err={mass_err:.2e}")

    mass_final = float(phi.sum()) * mesh.dx * mesh.dy
    print(f"[{label}] Done. Final mass_err = {abs(mass_final - mass0):.2e}\n")

    return phi_history, t_history


def run_rti_cdi(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[float]]:
    """Run the RTI simulation using CDI (Task 3 approach)."""
    return _run_rti(cfg, build_rhs_cdi, "CDI")


def run_rti_acdi(cfg: dict[str, Any]) -> tuple[list[np.ndarray], list[float]]:
    """Run the RTI simulation using ACDI (Task 4 approach)."""
    return _run_rti(cfg, build_rhs_acdi, "ACDI")
