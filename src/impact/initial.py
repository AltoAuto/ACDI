"""
impact/initial.py
-----------------
Initial conditions for the 2D droplet impact simulation.

Sets up a liquid drop falling toward a liquid pool:
  - Pool: phi=1 (liquid) between pool_bot and pool_y, tanh transitions at both
    boundaries so phi->0 at y=0.  This prevents the doubly-periodic domain from
    wrapping pool liquid (phi=1 at y=0) to the top boundary (y=Ly, phi=0) and
    generating a spurious large CSF force there.
  - Drop: phi=1 (liquid) inside a circle at (drop_cx, drop_cy)
  - Union: phi = max(phi_drop, phi_pool)
  - Velocity: v = -drop_U (uniform step) inside the drop, zero everywhere else
"""

import numpy as np
from core.mesh import Mesh


def drop_pool_ic(
    mesh: Mesh,
    cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (phi, u_cc, v_cc) for a drop above a liquid pool.

    Parameters
    ----------
    mesh : Mesh
        Computational mesh.
    cfg : dict
        Configuration dict with keys: eps_factor, drop_R, drop_cx, drop_cy,
        drop_U, pool_y.  Optional key: pool_bot (default 0.4).

    Returns
    -------
    phi : np.ndarray, shape (ny, nx)
        Initial volume fraction (1 = liquid, 0 = gas).
    u_cc : np.ndarray, shape (ny, nx)
        Initial x-velocity (zero everywhere).
    v_cc : np.ndarray, shape (ny, nx)
        Initial y-velocity (-drop_U inside drop, 0 elsewhere).
    """
    eps = cfg["eps_factor"] * mesh.dx

    XC, YC = mesh.XC, mesh.YC

    # --- Pool: phi=1 between pool_bot and pool_y ---
    # Tanh transitions at both the free surface (pool_y) and the pool bottom
    # (pool_bot).  This ensures phi->0 at y=0 so the doubly-periodic BC does
    # not create a spurious phi gradient at the top boundary (y=Ly).
    pool_bot = cfg.get("pool_bot", 0.4)
    dist_top = cfg["pool_y"] - YC           # >0 inside pool (below surface)
    dist_bot = YC - pool_bot                # >0 inside pool (above bottom)
    phi_pool = (0.5 * (1.0 + np.tanh(dist_top / (2.0 * eps)))
                * 0.5 * (1.0 + np.tanh(dist_bot / (2.0 * eps))))

    # --- Drop: phi=1 inside circle ---
    R  = cfg["drop_R"]
    cx = cfg["drop_cx"]
    cy = cfg["drop_cy"]
    dist_drop = R - np.sqrt((XC - cx) ** 2 + (YC - cy) ** 2)
    phi_drop = 0.5 * (1.0 + np.tanh(dist_drop / (2.0 * eps)))

    # --- Union: take max to merge both liquid regions ---
    phi = np.maximum(phi_drop, phi_pool)

    # --- Velocity: drop falls downward at drop_U ---
    u_cc = np.zeros_like(phi)
    v_cc = -cfg["drop_U"] * phi_drop

    return phi, u_cc, v_cc
