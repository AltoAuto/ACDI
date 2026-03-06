"""
core/initial_conditions.py
--------------------------
ME 5351 HW2 - Initial conditions for the volume-fraction field phi.

Two canonical initial conditions are required by the homework:

1. circular_drop
   A smoothed circular drop centred at (x0, y0) with radius R.
   The smooth Heaviside profile follows the hyperbolic-tangent form used
   in phase-field methods:

       phi(x, y) = 0.5 * (1 + tanh((R - r) / (2 * eps)))

   where r = sqrt((x - x0)^2 + (y - y0)^2) and eps is the interface
   half-thickness (typically 1-2 mesh spacings for CDI/ACDI).

2. square_drop (optional utility)
   A sharp-edged square, useful for checking dispersion errors.

Reference
---------
  Jain 2022, Section 4 (test cases).
"""

import numpy as np
from .mesh import Mesh


def circular_drop(
    mesh: Mesh,
    x0: float,
    y0: float,
    R: float,
    eps: float,
) -> np.ndarray:
    """Return initial volume-fraction field for a circular drop.

    Uses a hyperbolic-tangent profile to produce a smooth diffuse interface
    of half-thickness `eps` centred on the circle of radius `R`.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    x0 : float
        x-coordinate of the drop centre.
    y0 : float
        y-coordinate of the drop centre.
    R : float
        Radius of the drop.
    eps : float
        Interface half-thickness (controls diffuse-interface width).
        Typically set to 1-2 * max(dx, dy).

    Returns
    -------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.  phi ~ 1 inside the drop, phi ~ 0 outside.
    """
    # Minimum-image distance for periodic domains
    dx_arr = mesh.XC - x0
    dy_arr = mesh.YC - y0
    dx_arr = dx_arr - mesh.Lx * np.round(dx_arr / mesh.Lx)
    dy_arr = dy_arr - mesh.Ly * np.round(dy_arr / mesh.Ly)
    r = np.sqrt(dx_arr**2 + dy_arr**2)
    phi = 0.5 * (1.0 + np.tanh((R - r) / (2.0 * eps)))
    return phi


def square_drop(
    mesh: Mesh,
    x0: float,
    y0: float,
    half_width: float,
    eps: float,
) -> np.ndarray:
    """Return initial volume-fraction field for a smoothed square drop.

    Constructed as the product of four smoothed step functions (one per side)
    using hyperbolic-tangent profiles.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    x0, y0 : float
        Centre of the square.
    half_width : float
        Half side-length of the square.
    eps : float
        Interface half-thickness.

    Returns
    -------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    """
    h = half_width
    phi_right = 0.5 * (1.0 + np.tanh(( (x0 + h) - mesh.XC) / (2.0 * eps)))
    phi_left  = 0.5 * (1.0 + np.tanh((mesh.XC - (x0 - h)) / (2.0 * eps)))
    phi_top   = 0.5 * (1.0 + np.tanh(( (y0 + h) - mesh.YC) / (2.0 * eps)))
    phi_bot   = 0.5 * (1.0 + np.tanh((mesh.YC - (y0 - h)) / (2.0 * eps)))
    return phi_right * phi_left * phi_top * phi_bot
