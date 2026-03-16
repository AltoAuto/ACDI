"""
rti/initial.py
--------------
Initial condition for the Rayleigh-Taylor instability.

Heavy fluid (φ=1) sits above the interface, light fluid (φ=0) below.
A multi-mode cosine perturbation seeds the instability.
"""

import numpy as np
from core.mesh import Mesh


def mound_initial_condition(
    mesh: Mesh,
    A0: float,
    sigma: float,
    eps: float,
) -> np.ndarray:
    """Gaussian mound of heavy fluid pointing downward — the 'drop' IC.

    A single large-amplitude downward tongue of heavy fluid is placed at the
    centre of the domain.  It mimics a dense droplet beginning to fall through
    lighter fluid, creating the classic RTI mushroom-cap geometry.

    Interface:
        y_int(x) = Ly/2 - A0*exp(-((x - Lx/2)/sigma)^2)
                        + 0.01*A0*cos(4*pi*x/Lx)   (tiny symmetry-breaker)

    phi = 0.5*(1 + tanh((y - y_int(x)) / (2*eps)))

    Parameters
    ----------
    mesh  : Mesh
    A0    : float   mound amplitude [m]  — typically 0.15-0.20*Ly
    sigma : float   Gaussian half-width [m] — typically 0.10-0.15*Lx
    eps   : float   interface half-thickness
    """
    cx = mesh.Lx / 2.0
    gauss = np.exp(-((mesh.XC - cx) / sigma) ** 2)
    # Small mode-2 cosine breaks perfect left-right symmetry so the mushroom
    # cap eventually rolls up into spirals rather than staying symmetric.
    symmetry_break = 0.01 * A0 * np.cos(4.0 * np.pi * mesh.XC / mesh.Lx)
    y_int = mesh.Ly / 2.0 - A0 * gauss + symmetry_break
    return 0.5 * (1.0 + np.tanh((mesh.YC - y_int) / (2.0 * eps)))


def drop_initial_condition(
    mesh: Mesh,
    R: float,
    cx: float,
    cy: float,
    eps: float,
) -> np.ndarray:
    """Isolated circular heavy-fluid droplet (phi=1 inside, phi=0 outside).

    The drop is placed at (cx, cy) in an otherwise light-fluid domain.

    Physics notes
    -------------
    - The DROP'S BOTTOM SURFACE is a heavy-over-light (unstable) interface
      → RTI fingers grow DOWNWARD from the base of the drop.
    - The TOP SURFACE is light-over-heavy (stable) → stays smooth.
    - The bottom fingers look like tentacles descending from the drop,
      which visually resembles a heavy droplet falling and mushrooming.

    Note on translation: in a doubly-periodic Stokes domain the drop CANNOT
    translate (no mean flow).  What you see instead is the RTI fingers
    growing downward from the bottom surface while the top stays intact.
    Use a tall domain (Ly >= 2*Lx, --Ly 2.0) so the fingers have room
    to travel before the periodic boundary wraps them back.

    A small teardrop perturbation (+A0*cos(2*theta) at bottom) seeds the
    RTI immediately so the fingers appear quickly.

    Parameters
    ----------
    mesh : Mesh
    R    : float   droplet radius [m]
    cx   : float   x-centre of the droplet [m]
    cy   : float   y-centre of the droplet [m]
    eps  : float   interface half-thickness
    """
    dx = mesh.XC - cx
    dy = mesh.YC - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)  # 0=right, pi/2=up, -pi/2=down

    # Teardrop perturbation: extends the drop further downward at the base
    # and adds 2-mode ripple to seed multiple RTI fingers at the bottom.
    # sin(theta) = -1 at the very bottom of the drop.
    extend_bottom = -0.20 * R * (1.0 - np.sin(theta)) / 2.0  # more radius at bottom
    ripple        =  0.04 * R * np.cos(3.0 * theta)           # 3-finger RTI seed
    R_eff = R + extend_bottom + ripple

    return 0.5 * (1.0 + np.tanh((R_eff - r) / (2.0 * eps)))


def rti_initial_condition(
    mesh: Mesh,
    A0: float,
    n_modes: int,
    eps: float,
) -> np.ndarray:
    """Compute the RTI initial volume-fraction field.

    Interface location:
        y_int(x) = Ly/2 + A0 * sum_{n=1}^{n_modes} cos(2*pi*n*x / Lx)

    Phase field:
        phi(x, y) = 0.5 * (1 + tanh((y - y_int(x)) / (2*eps)))

    phi → 1 above the interface (heavy fluid), phi → 0 below (light fluid).

    Parameters
    ----------
    mesh : Mesh
        Computational mesh.
    A0 : float
        Perturbation amplitude [m].
    n_modes : int
        Number of cosine modes in the interface perturbation.
    eps : float
        Interface half-thickness parameter.

    Returns
    -------
    phi : np.ndarray, shape (ny, nx)
        Initial volume-fraction field.
    """
    XC = mesh.XC  # (ny, nx)
    YC = mesh.YC  # (ny, nx)

    # Interface position
    y_int = mesh.Ly / 2.0
    for n in range(1, n_modes + 1):
        y_int = y_int + A0 * np.cos(2.0 * np.pi * n * XC / mesh.Lx)

    phi = 0.5 * (1.0 + np.tanh((YC - y_int) / (2.0 * eps)))
    return phi
