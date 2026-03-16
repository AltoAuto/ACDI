"""
rti/velocity.py
---------------
Stokes velocity solver for the Rayleigh-Taylor instability.

Solves the quasi-static Stokes equation in Fourier space (no iteration):

    ν∇⁴ψ = −2·g·At · ∂φ/∂x

In Fourier space (doubly-periodic domain):

    û(k) =  g_eff · kx·ky / K⁴ · φ̂(k)
    v̂(k) = −g_eff · kx²  / K⁴ · φ̂(k)

where g_eff = 2·g·At/ν is the lumped growth-rate parameter,
K² = kx²+ky², K⁴ = (K²)², and û(0,0) = v̂(0,0) = 0 (no mean flow).
"""

import numpy as np
from core.mesh import Mesh


def stokes_velocity(
    phi: np.ndarray,
    mesh: Mesh,
    g_eff: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the Stokes RTI velocity field via FFT.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    mesh : Mesh
        Computational mesh (provides nx, ny, dx, dy).
    g_eff : float
        Lumped growth-rate parameter: g_eff = 2*g*At/nu  [1/(m*s)].

    Returns
    -------
    u_face : np.ndarray, shape (ny, nx+1)
        x-velocity at east/west faces (face-staggered).
    v_face : np.ndarray, shape (ny+1, nx)
        y-velocity at north/south faces (face-staggered).
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # Forward transform
    phi_hat = np.fft.rfft2(phi)  # shape (ny, nx//2+1)

    # Wavenumber arrays (physical frequencies × 2π)
    kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)   # shape (nx//2+1,)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)    # shape (ny,)

    # 2-D wavenumber grids: broadcast kx along columns, ky along rows
    KX, KY = np.meshgrid(kx, ky)   # both (ny, nx//2+1)

    K2 = KX**2 + KY**2
    K4 = K2**2
    K4[0, 0] = 1.0  # avoid division by zero; mean flow is set to zero below

    # Velocity in Fourier space
    u_hat = g_eff * KX * KY / K4 * phi_hat
    v_hat = -g_eff * KX**2 / K4 * phi_hat

    # Zero mean flow
    u_hat[0, 0] = 0.0
    v_hat[0, 0] = 0.0

    # Inverse transform to cell centres
    u_cc = np.fft.irfft2(u_hat, s=(ny, nx))  # (ny, nx)
    v_cc = np.fft.irfft2(v_hat, s=(ny, nx))  # (ny, nx)

    # Interpolate cell-centre velocities to faces (periodic averaging)
    # u_face[:, i] = 0.5*(u_cc[:, i-1] + u_cc[:, i])  with periodic wrap
    u_face = np.empty((ny, nx + 1))
    for i in range(nx + 1):
        u_face[:, i] = 0.5 * (u_cc[:, (i - 1) % nx] + u_cc[:, i % nx])

    # v_face[j, :] = 0.5*(v_cc[j-1, :] + v_cc[j, :])  with periodic wrap
    v_face = np.empty((ny + 1, nx))
    for j in range(ny + 1):
        v_face[j, :] = 0.5 * (v_cc[(j - 1) % ny, :] + v_cc[j % ny, :])

    return u_face, v_face
