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
    """Return the CDI regularisation RHS term (Jain 2022, Eq. 21).

    Implements:
        R_CDI = Gamma * div[ eps*grad(phi) - phi*(1-phi)*n_hat ]

    Discretisation follows Jain 2022 Eq. (21): the interface normal n_hat is
    computed at cell centres using 3-point central differences in both x and y,
    then averaged arithmetically to each face (the overbar in Eq. 21 denotes
    this arithmetic average).  The diffusion term uses the compact 2-cell face
    gradient (Delta_j phi = phi_{m+1} - phi_m).

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
    dx, dy = mesh.dx, mesh.dy

    # Clip phi to [0,1] for sharpening amplitude (phi*(1-phi) must stay >= 0).
    # Gradients for n_hat use raw phi so interface normals are unaffected.
    phi_c = np.clip(phi, 0.0, 1.0)

    # Cell-centre unit normals from 3-point central differences (both directions)
    dphi_dx_cc = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dx)
    dphi_dy_cc = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dy)
    mag_cc = np.sqrt(dphi_dx_cc**2 + dphi_dy_cc**2) + 1e-14
    nx_hat_cc = dphi_dx_cc / mag_cc
    ny_hat_cc = dphi_dy_cc / mag_cc

    # x-face fluxes: A_x at face (i+1/2) between cells i and i+1
    # n_hat at face = arithmetic average of CC normals (Eq. 21 overbar notation)
    phi_r = np.roll(phi_c, -1, axis=1)
    dphi_dx_f = (np.roll(phi, -1, axis=1) - phi) / dx
    phi_bar = 0.5 * (phi_c + phi_r)
    nx_hat_f = 0.5 * (nx_hat_cc + np.roll(nx_hat_cc, -1, axis=1))
    Ax = eps * dphi_dx_f - phi_bar * (1.0 - phi_bar) * nx_hat_f

    # y-face fluxes: A_y at face (j+1/2) between rows j and j+1
    phi_t = np.roll(phi_c, -1, axis=0)
    dphi_dy_f = (np.roll(phi, -1, axis=0) - phi) / dy
    phi_bar = 0.5 * (phi_c + phi_t)
    ny_hat_f = 0.5 * (ny_hat_cc + np.roll(ny_hat_cc, -1, axis=0))
    Ay = eps * dphi_dy_f - phi_bar * (1.0 - phi_bar) * ny_hat_f

    # Divergence: (A_{i+1/2} - A_{i-1/2}) / dx, periodic via roll
    rhs = Gamma * (
        (Ax - np.roll(Ax, 1, axis=1)) / dx +
        (Ay - np.roll(Ay, 1, axis=0)) / dy
    )
    return rhs


def acdi_regularization(
    phi: np.ndarray,
    mesh: Mesh,
    eps: float,
    Gamma: float = 1.0,
) -> np.ndarray:
    """Return the ACDI regularisation RHS term (Jain 2022, Eq. 21).

    Implements (Eq. 7):
        R_ACDI = Gamma * div[ eps*grad(phi)
                              - (1/4)(1 - tanh^2(psi/(2*eps))) * grad(psi)/|grad(psi)| ]

    where psi = eps * ln(phi / (1 - phi)) is the signed-distance-like variable.

    Discretisation follows Jain 2022 Eq. (21) — the second-order flux-split
    central scheme used in the paper's main results (Tables 1 and 3):

        A_j|(m+1/2) = Gamma * { eps/dx*(phi_{m+1}-phi_m)
                                - (1/4)*(1-tanh^2(psi_bar/(2*eps))) * n_hat_face }

    where psi_bar = (psi_m + psi_{m+1})/2 (face-averaged psi, then tanh applied)
    and n_hat_face = (n_hat_cc_m + n_hat_cc_{m+1})/2 (CC normals averaged to face).
    Both averages follow the overbar notation defined in Eq. (21).

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    mesh : Mesh
        Computational mesh.
    eps : float
        Interface half-thickness parameter.
    Gamma : float
        Regularisation rate parameter (>= |u|_max for ACDI, Eq. 9).

    Returns
    -------
    rhs_reg : np.ndarray, shape (ny, nx)
        ACDI regularisation contribution to d(phi)/dt.
    """
    dx, dy = mesh.dx, mesh.dy

    # Signed-distance-like variable: psi = eps * ln((phi+w) / (1-phi+w))
    # Clip phi to [0,1] first (paper Sec. 3.1), small w avoids log(0).
    w = 1e-100
    phi_c = np.clip(phi, 0.0, 1.0)
    psi = eps * np.log((phi_c + w) / (1.0 - phi_c + w))

    # Cell-centre unit normals from psi (3-point central differences)
    dpsi_dx_cc = (np.roll(psi, -1, axis=1) - np.roll(psi, 1, axis=1)) / (2.0 * dx)
    dpsi_dy_cc = (np.roll(psi, -1, axis=0) - np.roll(psi, 1, axis=0)) / (2.0 * dy)
    mag_psi_cc = np.sqrt(dpsi_dx_cc**2 + dpsi_dy_cc**2) + 1e-14
    nx_hat_cc = dpsi_dx_cc / mag_psi_cc
    ny_hat_cc = dpsi_dy_cc / mag_psi_cc

    # x-face fluxes: A_x at face (i+1/2) between cells i and i+1 (Eq. 21)
    psi_r = np.roll(psi, -1, axis=1)
    psi_bar = 0.5 * (psi + psi_r)                               # face-averaged psi
    c_x = 0.25 * (1.0 - np.tanh(psi_bar / (2.0 * eps)) ** 2)   # sharpening coeff
    nx_hat_f = 0.5 * (nx_hat_cc + np.roll(nx_hat_cc, -1, axis=1))  # face-averaged n_hat
    dphi_dx_f = (np.roll(phi, -1, axis=1) - phi) / dx
    Ax = eps * dphi_dx_f - c_x * nx_hat_f

    # y-face fluxes: A_y at face (j+1/2) between rows j and j+1 (Eq. 21)
    psi_t = np.roll(psi, -1, axis=0)
    psi_bar = 0.5 * (psi + psi_t)
    c_y = 0.25 * (1.0 - np.tanh(psi_bar / (2.0 * eps)) ** 2)
    ny_hat_f = 0.5 * (ny_hat_cc + np.roll(ny_hat_cc, -1, axis=0))
    dphi_dy_f = (np.roll(phi, -1, axis=0) - phi) / dy
    Ay = eps * dphi_dy_f - c_y * ny_hat_f

    # Divergence: (A_{i+1/2} - A_{i-1/2}) / dx, periodic via roll
    rhs = Gamma * (
        (Ax - np.roll(Ax, 1, axis=1)) / dx +
        (Ay - np.roll(Ay, 1, axis=0)) / dy
    )
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
