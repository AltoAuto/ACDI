"""
core/time_integration.py
------------------------
ME 5351 HW2 - Time integration routines for explicit ODE advancement.

The semi-discrete system after spatial discretisation is:

    d(phi)/dt = L(phi, t)

where L is the spatial operator (advection +/- regularisation terms).
Two integrators are provided:

euler_step
    First-order explicit (forward) Euler:
        phi^{n+1} = phi^n + dt * L(phi^n, t^n)
    Used in Tasks 1 and 2.

rk4_step
    Classical 4-stage, 4th-order Runge-Kutta (RK4):
        k1 = L(phi^n,             t^n)
        k2 = L(phi^n + dt/2 * k1, t^n + dt/2)
        k3 = L(phi^n + dt/2 * k2, t^n + dt/2)
        k4 = L(phi^n + dt   * k3, t^n + dt)
        phi^{n+1} = phi^n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    Used in Task 3 and Task 4.

Notes
-----
- The operator `L` is passed as a callable `rhs_fn(phi, t) -> np.ndarray`.
  This keeps the time integrators decoupled from the specific spatial scheme
  and regularisation used in each task.
- Stability requires CFL <= 1 for Euler; RK4 has a larger stability region
  (CFL_max ~ 2.8 for pure advection) but is still conditionally stable.
"""

from typing import Callable
import numpy as np


def euler_step(
    phi: np.ndarray,
    t: float,
    dt: float,
    rhs_fn: Callable[[np.ndarray, float], np.ndarray],
) -> np.ndarray:
    """Advance phi by one time step using first-order explicit Euler.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field at time t^n.
    t : float
        Current simulation time t^n.
    dt : float
        Time step size.
    rhs_fn : callable
        Function with signature (phi, t) -> rhs of shape (ny, nx).
        Should return  L(phi, t) = -div(u*phi) + regularisation terms.

    Returns
    -------
    phi_new : np.ndarray, shape (ny, nx)
        Volume-fraction field at time t^{n+1} = t^n + dt.
    """
    return phi + dt * rhs_fn(phi, t)


def rk4_step(
    phi: np.ndarray,
    t: float,
    dt: float,
    rhs_fn: Callable[[np.ndarray, float], np.ndarray],
) -> np.ndarray:
    """Advance phi by one time step using classical 4th-order Runge-Kutta.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field at time t^n.
    t : float
        Current simulation time t^n.
    dt : float
        Time step size.
    rhs_fn : callable
        Function with signature (phi, t) -> rhs of shape (ny, nx).
        Note: for time-dependent velocity fields (shear flow test case),
        the callable must correctly evaluate the velocity at the intermediate
        time levels t + dt/2 and t + dt.

    Returns
    -------
    phi_new : np.ndarray, shape (ny, nx)
        Volume-fraction field at time t^{n+1} = t^n + dt.
    """
    k1 = rhs_fn(phi, t)
    k2 = rhs_fn(phi + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs_fn(phi + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs_fn(phi + dt * k3, t + dt)
    return phi + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def compute_cfl(
    u_face: np.ndarray,
    v_face: np.ndarray,
    dt: float,
    dx: float,
    dy: float,
) -> float:
    """Compute the maximum CFL number over the domain.

    CFL = max(|u| / dx + |v| / dy) * dt

    Parameters
    ----------
    u_face : np.ndarray
        x-velocity at east/west faces.
    v_face : np.ndarray
        y-velocity at north/south faces.
    dt : float
        Proposed time step size.
    dx, dy : float
        Mesh spacings.

    Returns
    -------
    cfl : float
        Maximum CFL number (dimensionless).
    """
    max_u = np.max(np.abs(u_face))
    max_v = np.max(np.abs(v_face))
    return (max_u / dx + max_v / dy) * dt
