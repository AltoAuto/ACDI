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

ACDI (Accurate CDI) -- Jain 2022, Eq. (7), (20)-(21)
----------------------------------------------------
ACDI replaces the phi-based sharpening flux with a psi-based form:

    R_ACDI = Gamma * div[ eps*grad(phi)
                          - (1/4)(1 - tanh^2(psi/(2*eps))) * grad(psi)/|grad(psi)| ]

where psi = eps * ln(phi / (1-phi)) is a signed-distance-like variable.
The diffusion coefficient is the SAME as CDI (full eps).  The improvement
comes from using psi -- a smoother function than phi -- to compute the
sharpening flux and normal, yielding more accurate discrete gradients.
Gamma >= |u|_max for ACDI (Eq. 9).

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
    """Return the CDI regularisation RHS term (Jain 2022, Eq. 1/21).

    Implements:
        R_CDI = Gamma * div[ eps*grad(phi) - phi*(1-phi)*n_hat ]

    Discretisation follows Jain 2022 Eq. (21): diffusion uses compact face
    gradients, sharpening flux (phi*(1-phi)*n_hat) is computed at cell centres
    and averaged to faces.  This ensures better cancellation between the
    diffusion and sharpening terms at the discrete level.

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

    # Cell-centre gradients and normalised normal (periodic)
    dphi_dx_cc = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dx)
    dphi_dy_cc = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dy)
    mag_cc = np.sqrt(dphi_dx_cc**2 + dphi_dy_cc**2) + 1e-14

    # Cell-centre sharpening vector: phi*(1-phi)*n_hat
    sharp_x_cc = phi * (1.0 - phi) * dphi_dx_cc / mag_cc
    sharp_y_cc = phi * (1.0 - phi) * dphi_dy_cc / mag_cc

    # --- x-faces (nx+1 faces) ---
    Ax = np.empty((ny_c, nx_c + 1))
    for i in range(nx_c + 1):
        il = (i - 1) % nx_c
        ir = i % nx_c
        dphi_dx_f = (phi[:, ir] - phi[:, il]) / dx
        Ax[:, i] = eps * dphi_dx_f - 0.5 * (sharp_x_cc[:, il] + sharp_x_cc[:, ir])

    # --- y-faces (ny+1 faces) ---
    Ay = np.empty((ny_c + 1, nx_c))
    for j in range(ny_c + 1):
        jb = (j - 1) % ny_c
        jt = j % ny_c
        dphi_dy_f = (phi[jt, :] - phi[jb, :]) / dy
        Ay[j, :] = eps * dphi_dy_f - 0.5 * (sharp_y_cc[jb, :] + sharp_y_cc[jt, :])

    # Divergence of the regularisation flux, scaled by Gamma
    rhs = Gamma * ((Ax[:, 1:] - Ax[:, :-1]) / dx + (Ay[1:, :] - Ay[:-1, :]) / dy)
    return rhs


def acdi_regularization(
    phi: np.ndarray,
    mesh: Mesh,
    eps: float,
    Gamma: float = 1.0,
) -> np.ndarray:
    """Return the ACDI regularisation RHS term (Jain 2022, Eq. 7/21).

    Implements (Eq. 7):
        R_ACDI = Gamma * div[ eps*grad(phi)
                              - (1/4)(1 - tanh^2(psi/(2*eps))) * grad(psi)/|grad(psi)| ]

    where psi = eps * ln(phi / (1 - phi)) is the signed-distance-like variable.
    The diffusion uses full eps (same as CDI).  The sharpening flux uses
    the psi-based form for improved discrete accuracy (psi is smoother
    than phi, yielding more accurate gradient/normal computations).

    Discretisation follows Jain 2022 Eq. (21): diffusion uses compact face
    gradients, sharpening flux (coeff*n_hat) is computed at cell centres
    and averaged to faces.

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

    # Signed-distance-like variable: psi = eps * ln((phi+w) / (1-phi+w))
    # Clip phi to [0,1] first (paper Sec. 3.1), then add w per Eq. (8).
    w = 1e-100
    phi_c = np.clip(phi, 0.0, 1.0)
    psi = eps * np.log((phi_c + w) / (1.0 - phi_c + w))

    # Cell-centre gradients of psi and normalised normal
    dpsi_dx_cc = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * dx)
    dpsi_dy_cc = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * dy)
    mag_psi_cc = np.sqrt(dpsi_dx_cc**2 + dpsi_dy_cc**2) + 1e-14

    # Cell-centre sharpening vector: coeff * n_hat (Eq. 21)
    coeff_cc = 0.25 * (1.0 - np.tanh(psi / (2.0 * eps)) ** 2)
    sharp_x_cc = coeff_cc * dpsi_dx_cc / mag_psi_cc
    sharp_y_cc = coeff_cc * dpsi_dy_cc / mag_psi_cc

    # --- x-faces ---
    Ax = np.empty((ny_c, nx_c + 1))
    for i in range(nx_c + 1):
        il = (i - 1) % nx_c
        ir = i % nx_c
        dphi_dx_f = (phi[:, ir] - phi[:, il]) / dx
        Ax[:, i] = eps * dphi_dx_f - 0.5 * (sharp_x_cc[:, il] + sharp_x_cc[:, ir])

    # --- y-faces ---
    Ay = np.empty((ny_c + 1, nx_c))
    for j in range(ny_c + 1):
        jb = (j - 1) % ny_c
        jt = j % ny_c
        dphi_dy_f = (phi[jt, :] - phi[jb, :]) / dy
        Ay[j, :] = eps * dphi_dy_f - 0.5 * (sharp_y_cc[jb, :] + sharp_y_cc[jt, :])

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
