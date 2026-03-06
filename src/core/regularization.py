"""
core/regularization.py
----------------------
ME 5351 HW2 - Interface regularisation kernels for CDI and ACDI methods.

Phase-field interface methods require a sharpening / regularisation term
added to the scalar transport equation to counteract numerical diffusion
and maintain a prescribed diffuse-interface profile.

CDI (Conservative Diffuse Interface) -- Jain 2022, Eq. (6)-(8)
---------------------------------------------------------------
The CDI term takes the form:

    R_CDI = Gamma * [ eps * lap(phi) - div( phi*(1-phi)*n_hat ) ]

where n_hat = grad(phi) / |grad(phi)| is the interface unit normal,
eps is the interface half-thickness parameter, and Gamma is a rate
parameter that must satisfy Gamma >= |u|_max / (2*eps/dx - 1).

ACDI (Accurate CDI) -- Jain 2022, Eq. (20)-(21)
-------------------------------------------------
ACDI uses a modified regularisation with the diffusion coefficient halved:

    R_ACDI = Gamma * [ 0.5 * eps * lap(phi) - div( phi*(1-phi)*n_hat ) ]

paired with skew-symmetric advection splitting.
Gamma >= |u|_max for ACDI.

Both methods preserve:
  - Global mass (integral of phi) to machine precision.
  - The tanh interface profile in steady state (no advection).

Reference
---------
  Jain, S. S. (2022). J. Comput. Phys., 469, 111529.
  Eqs. (6), (7), (8), (20), (21).
"""

import numpy as np
from .mesh import Mesh


def compute_interface_normal(
    phi: np.ndarray,
    mesh: Mesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the unit interface normal n_hat = grad(phi) / |grad(phi)|.

    Uses 2nd-order central differences for the gradient.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    mesh : Mesh
        Computational mesh (provides dx, dy and periodic roll helpers).

    Returns
    -------
    nx_hat : np.ndarray, shape (ny, nx)
        x-component of the unit interface normal.
    ny_hat : np.ndarray, shape (ny, nx)
        y-component of the unit interface normal.
    """
    dphi_dx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * mesh.dx)
    dphi_dy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * mesh.dy)

    mag = np.sqrt(dphi_dx**2 + dphi_dy**2) + 1e-14
    return dphi_dx / mag, dphi_dy / mag


def cdi_regularization(
    phi: np.ndarray,
    mesh: Mesh,
    eps: float,
    Gamma: float = 1.0,
) -> np.ndarray:
    """Return the CDI regularisation RHS term.

    Implements:
        R_CDI = Gamma * [ eps * lap(phi) - div( phi*(1-phi)*n_hat ) ]

    Computed at faces in divergence form for exact conservation:
        R = Gamma * div[ eps*grad(phi) - phi*(1-phi)*n_hat ]

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    mesh : Mesh
        Computational mesh.
    eps : float
        Interface half-thickness parameter.
    Gamma : float
        Regularisation rate parameter.

    Returns
    -------
    rhs_reg : np.ndarray, shape (ny, nx)
        Regularisation contribution to d(phi)/dt.
    """
    nx_c, ny_c = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # Cell-centre gradients (periodic)
    dphi_dy_cc = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dy)
    dphi_dx_cc = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dx)

    # --- x-faces (nx+1 faces) ---
    Ax = np.empty((ny_c, nx_c + 1))
    for i in range(nx_c + 1):
        il = (i - 1) % nx_c
        ir = i % nx_c
        phi_f = 0.5 * (phi[:, il] + phi[:, ir])
        dphi_dx_f = (phi[:, ir] - phi[:, il]) / dx
        dphi_dy_f = 0.5 * (dphi_dy_cc[:, il] + dphi_dy_cc[:, ir])
        mag_f = np.sqrt(dphi_dx_f**2 + dphi_dy_f**2) + 1e-14
        # Flux: eps*dphi_dx - phi*(1-phi)*n_x = eps*dphi_dx - phi*(1-phi)*dphi_dx/|grad|
        Ax[:, i] = eps * dphi_dx_f - phi_f * (1.0 - phi_f) * dphi_dx_f / mag_f

    # --- y-faces (ny+1 faces) ---
    Ay = np.empty((ny_c + 1, nx_c))
    for j in range(ny_c + 1):
        jb = (j - 1) % ny_c
        jt = j % ny_c
        phi_f = 0.5 * (phi[jb, :] + phi[jt, :])
        dphi_dy_f = (phi[jt, :] - phi[jb, :]) / dy
        dphi_dx_f = 0.5 * (dphi_dx_cc[jb, :] + dphi_dx_cc[jt, :])
        mag_f = np.sqrt(dphi_dx_f**2 + dphi_dy_f**2) + 1e-14
        Ay[j, :] = eps * dphi_dy_f - phi_f * (1.0 - phi_f) * dphi_dy_f / mag_f

    # Divergence of the regularisation flux, scaled by Gamma
    rhs = Gamma * ((Ax[:, 1:] - Ax[:, :-1]) / dx + (Ay[1:, :] - Ay[:-1, :]) / dy)
    return rhs


def acdi_regularization(
    phi: np.ndarray,
    mesh: Mesh,
    eps: float,
    Gamma: float = 1.0,
) -> np.ndarray:
    """Return the ACDI regularisation RHS term (Jain 2022, Eq. 20-21).

    Implements:
        R_ACDI = Gamma * [ 0.5*eps*lap(phi) - div( phi*(1-phi)*n_hat ) ]

    In flux form:
        R = Gamma * div[ 0.5*eps*grad(phi) - phi*(1-phi)*n_hat ]

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    mesh : Mesh
        Computational mesh.
    eps : float
        Interface half-thickness parameter.
    Gamma : float
        Regularisation rate parameter (>= |u|_max for ACDI).

    Returns
    -------
    rhs_reg : np.ndarray, shape (ny, nx)
        ACDI regularisation contribution to d(phi)/dt.
    """
    nx_c, ny_c = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    dphi_dy_cc = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dy)
    dphi_dx_cc = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dx)

    # --- x-faces ---
    Ax = np.empty((ny_c, nx_c + 1))
    for i in range(nx_c + 1):
        il = (i - 1) % nx_c
        ir = i % nx_c
        phi_f = 0.5 * (phi[:, il] + phi[:, ir])
        dphi_dx_f = (phi[:, ir] - phi[:, il]) / dx
        dphi_dy_f = 0.5 * (dphi_dy_cc[:, il] + dphi_dy_cc[:, ir])
        mag_f = np.sqrt(dphi_dx_f**2 + dphi_dy_f**2) + 1e-14
        # ACDI flux: 0.5*eps*dphi_dx - phi*(1-phi)*n_x
        Ax[:, i] = 0.5 * eps * dphi_dx_f - phi_f * (1.0 - phi_f) * dphi_dx_f / mag_f

    # --- y-faces ---
    Ay = np.empty((ny_c + 1, nx_c))
    for j in range(ny_c + 1):
        jb = (j - 1) % ny_c
        jt = j % ny_c
        phi_f = 0.5 * (phi[jb, :] + phi[jt, :])
        dphi_dy_f = (phi[jt, :] - phi[jb, :]) / dy
        dphi_dx_f = 0.5 * (dphi_dx_cc[jb, :] + dphi_dx_cc[jt, :])
        mag_f = np.sqrt(dphi_dx_f**2 + dphi_dy_f**2) + 1e-14
        Ay[j, :] = 0.5 * eps * dphi_dy_f - phi_f * (1.0 - phi_f) * dphi_dy_f / mag_f

    rhs = Gamma * ((Ax[:, 1:] - Ax[:, :-1]) / dx + (Ay[1:, :] - Ay[:-1, :]) / dy)
    return rhs


def laplacian(
    phi: np.ndarray,
    mesh: Mesh,
) -> np.ndarray:
    """Compute the discrete Laplacian using 2nd-order central differences.

    lap(phi)_ij = (phi_{i+1,j} - 2*phi_{i,j} + phi_{i-1,j}) / dx^2
                + (phi_{i,j+1} - 2*phi_{i,j} + phi_{i,j-1}) / dy^2

    Periodic BCs applied via numpy.roll.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Scalar field.
    mesh : Mesh
        Computational mesh.

    Returns
    -------
    lap_phi : np.ndarray, shape (ny, nx)
        Discrete Laplacian of phi.
    """
    lap_x = (np.roll(phi, -1, axis=1) - 2.0 * phi + np.roll(phi, 1, axis=1)) / mesh.dx**2
    lap_y = (np.roll(phi, -1, axis=0) - 2.0 * phi + np.roll(phi, 1, axis=0)) / mesh.dy**2
    return lap_x + lap_y
