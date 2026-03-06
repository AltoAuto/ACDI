"""
core/flux_schemes.py
--------------------
ME 5351 HW2 - Spatial discretisation schemes for the advective flux.

This module collects the convective flux assembly routines used across tasks.
All functions operate on face-staggered velocity arrays and return the net
flux divergence for each cell (i.e., the right-hand side of the transport
equation BEFORE time integration).

Schemes implemented
-------------------
upwind_flux
    First-order upwind (donor-cell) scheme.  Numerically diffusive but
    unconditionally bounded (0 <= phi <= 1).  Used in Tasks 1 and 2.

central_flux
    Second-order central (arithmetic average) face interpolation.
    Used in Task 3 (CDI with 2nd-order space) and as the base for the
    skew-symmetric splitting in Task 4 (ACDI).

divergence_rhs
    Assembles the full divergence term  -div(u * phi)  from a given face
    flux scheme.  Returns the volumetric RHS array ddt(phi) contribution
    from advection.

Notes
-----
- Periodic boundary conditions are handled via numpy.roll.
- Face velocities must be provided on the staggered grid (see velocity_fields.py).
- The CFL number should satisfy CFL = |u| * dt / dx <= 1 for explicit Euler
  (Tasks 1-2); RK4 allows somewhat larger values but stability is not
  unconditional.

Reference
---------
  Jain 2022, Eq. (3)-(4) for the scalar transport equation.
"""

import numpy as np
from .mesh import Mesh


def upwind_flux(
    phi: np.ndarray,
    u_face: np.ndarray,
    v_face: np.ndarray,
    mesh: Mesh,
) -> np.ndarray:
    """Compute the advective flux divergence using first-order upwind.

    The upwind (donor-cell) face value is:
        phi_face = phi_upwind_cell  (based on sign of normal velocity)

    The flux divergence returned is  F = -div(u * phi)  per unit volume,
    ready to be used as the RHS for explicit time integration.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Current volume-fraction field.
    u_face : np.ndarray, shape (ny, nx+1)
        x-velocity at east/west faces.
    v_face : np.ndarray, shape (ny+1, nx)
        y-velocity at north/south faces.
    mesh : Mesh
        The computational mesh (provides dx, dy).

    Returns
    -------
    rhs : np.ndarray, shape (ny, nx)
        Advective RHS  -div(u * phi).
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # --- x-direction fluxes at vertical faces ---
    # Face i sits between cell i-1 (left) and cell i (right), periodic
    # phi_left[j, i] = phi[j, (i-1) % nx],  phi_right[j, i] = phi[j, i % nx]
    # We have nx+1 faces; face 0 and face nx are the same (periodic).
    phi_left = np.empty((ny, nx + 1))
    phi_right = np.empty((ny, nx + 1))
    # Interior faces 0..nx: left cell is (i-1)%nx, right cell is i%nx
    for i in range(nx + 1):
        phi_left[:, i] = phi[:, (i - 1) % nx]
        phi_right[:, i] = phi[:, i % nx]

    # Upwind selection
    phi_face_x = np.where(u_face >= 0, phi_left, phi_right)
    flux_x = u_face * phi_face_x  # (ny, nx+1)

    # --- y-direction fluxes at horizontal faces ---
    phi_bot = np.empty((ny + 1, nx))
    phi_top = np.empty((ny + 1, nx))
    for j in range(ny + 1):
        phi_bot[j, :] = phi[(j - 1) % ny, :]
        phi_top[j, :] = phi[j % ny, :]

    phi_face_y = np.where(v_face >= 0, phi_bot, phi_top)
    flux_y = v_face * phi_face_y  # (ny+1, nx)

    # Divergence: -[ (flux_east - flux_west)/dx + (flux_north - flux_south)/dy ]
    # For cell (j, i): east face = i+1, west face = i
    #                   north face = j+1, south face = j
    div_x = (flux_x[:, 1:] - flux_x[:, :-1]) / dx   # (ny, nx)
    div_y = (flux_y[1:, :] - flux_y[:-1, :]) / dy    # (ny, nx)

    return -(div_x + div_y)


def central_flux(
    phi: np.ndarray,
    u_face: np.ndarray,
    v_face: np.ndarray,
    mesh: Mesh,
) -> np.ndarray:
    """Compute the advective flux divergence using 2nd-order central differencing.

    The face value is the arithmetic average of the two neighbouring cells:
        phi_face = 0.5 * (phi_L + phi_R)

    Note: this scheme is non-dissipative and may produce oscillations near
    sharp gradients.  It is paired with CDI/ACDI regularisation in Tasks 3-4
    to maintain boundedness.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Current volume-fraction field.
    u_face : np.ndarray, shape (ny, nx+1)
        x-velocity at east/west faces.
    v_face : np.ndarray, shape (ny+1, nx)
        y-velocity at north/south faces.
    mesh : Mesh
        The computational mesh.

    Returns
    -------
    rhs : np.ndarray, shape (ny, nx)
        Advective RHS  -div(u * phi).
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # x-direction: face value = average of left and right cells
    phi_left = np.empty((ny, nx + 1))
    phi_right = np.empty((ny, nx + 1))
    for i in range(nx + 1):
        phi_left[:, i] = phi[:, (i - 1) % nx]
        phi_right[:, i] = phi[:, i % nx]

    phi_face_x = 0.5 * (phi_left + phi_right)
    flux_x = u_face * phi_face_x

    # y-direction
    phi_bot = np.empty((ny + 1, nx))
    phi_top = np.empty((ny + 1, nx))
    for j in range(ny + 1):
        phi_bot[j, :] = phi[(j - 1) % ny, :]
        phi_top[j, :] = phi[j % ny, :]

    phi_face_y = 0.5 * (phi_bot + phi_top)
    flux_y = v_face * phi_face_y

    div_x = (flux_x[:, 1:] - flux_x[:, :-1]) / dx
    div_y = (flux_y[1:, :] - flux_y[:-1, :]) / dy

    return -(div_x + div_y)


def divergence_rhs(
    phi: np.ndarray,
    u_face: np.ndarray,
    v_face: np.ndarray,
    mesh: Mesh,
    scheme: str = "upwind",
) -> np.ndarray:
    """Dispatch to the requested flux scheme and return the flux divergence RHS.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    u_face, v_face : np.ndarray
        Face-staggered velocity components.
    mesh : Mesh
        Computational mesh.
    scheme : {'upwind', 'central'}
        Name of the spatial scheme to use.

    Returns
    -------
    rhs : np.ndarray, shape (ny, nx)
        -div(u * phi) evaluated with the chosen scheme.
    """
    if scheme == "upwind":
        return upwind_flux(phi, u_face, v_face, mesh)
    elif scheme == "central":
        return central_flux(phi, u_face, v_face, mesh)
    else:
        raise ValueError(f"Unknown flux scheme: {scheme!r}")
