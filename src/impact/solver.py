"""
impact/solver.py
----------------
Time-stepping loop for the 2D droplet impact simulation.

Calls step_ns at each iteration and stores snapshots every save_freq steps.
Prints stability diagnostics at startup and progress every 10% of t_end.
"""

import math
import numpy as np

from core.mesh import Mesh
from impact.initial import drop_pool_ic
from impact.ns_solver import step_ns, _build_wavenumbers


def run_impact(cfg: dict) -> list[dict]:
    """Run the droplet impact simulation.

    Parameters
    ----------
    cfg : dict
        Configuration dict (see impact/config.py for keys).

    Returns
    -------
    history : list of dict
        Each entry has keys: phi, u, v, p, t.
    """
    # Build mesh
    nx   = cfg["nx"]
    ny   = cfg.get("ny", nx)
    mesh = Mesh(nx=nx, ny=ny, Lx=cfg["Lx"], Ly=cfg["Ly"])
    dx   = mesh.dx

    # Pre-build wavenumber matrix (reused every step)
    K2 = _build_wavenumbers(mesh)

    # Initial conditions
    phi, u, v = drop_pool_ic(mesh, cfg)

    # Buoyancy reference density (pool + gas only, NO drop).
    #
    # rho_init must represent the "background" state before the drop was placed:
    #   - Pool region: rho_init = rho1  → g_body = 0 (pool in equilibrium)
    #   - Gas region (incl. drop location): rho_init = rho2  → g_body ≈ -g (drop falls)
    #
    # Using the full initial phi (which includes the drop) would make
    # rho_init = rho1 at the drop location, cancelling g_body there and
    # preventing the drop from falling.  Instead we recompute phi_pool_only
    # (the pool IC without the drop) to get the correct reference.
    rho1_c = cfg["rho1"]; rho2_c = cfg["rho2"]
    eps_ref   = cfg["eps_factor"] * mesh.dx
    pool_bot  = cfg.get("pool_bot", 0.4)
    dist_top  = cfg["pool_y"] - mesh.YC
    dist_bot  = mesh.YC - pool_bot
    phi_pool_only = (0.5 * (1.0 + np.tanh(dist_top / (2.0 * eps_ref)))
                     * 0.5 * (1.0 + np.tanh(dist_bot / (2.0 * eps_ref))))
    cfg["rho_init"] = rho1_c * phi_pool_only + rho2_c * (1.0 - phi_pool_only)

    dt        = cfg["dt"]
    t_end     = cfg["t_end"]
    save_freq = cfg.get("save_freq", 10)

    # -----------------------------------------------------------------
    # Stability diagnostics
    # -----------------------------------------------------------------
    rho1  = cfg["rho1"]
    rho2  = cfg["rho2"]
    mu1   = cfg["mu1"]
    sigma = cfg["sigma"]
    U0    = cfg["drop_U"]

    dt_cfl  = dx / max(U0, 1e-6)
    dt_visc = rho1 * dx ** 2 / (2.0 * mu1)
    dt_cap  = math.sqrt(rho2 * dx ** 3 / (2.0 * math.pi * sigma))
    dt_min  = min(dt_cfl, dt_visc, dt_cap)

    print("=" * 64)
    print("Droplet Impact - Stability Diagnostics")
    print("=" * 64)
    print(f"  Grid       : {nx} x {ny}  dx={dx:.4e}  dy={mesh.dy:.4e}")
    print(f"  dt (used)  : {dt:.2e}")
    print(f"  dt_CFL     : {dt_cfl:.2e}   (dx / U0)")
    print(f"  dt_visc    : {dt_visc:.2e}   (rho1*dx^2 / 2*mu1)")
    print(f"  dt_cap     : {dt_cap:.2e}   (sqrt(rho2*dx^3 / 2pi*sigma))")
    if dt > dt_min:
        print(f"  WARNING: dt={dt:.2e} exceeds stability limit {dt_min:.2e}!")
    else:
        print(f"  dt < dt_min={dt_min:.2e}  [OK] stable")
    print("=" * 64)

    mass0 = float(phi.sum()) * mesh.dx * mesh.dy
    print(f"Initial phi  : [{phi.min():.4f}, {phi.max():.4f}]  mass={mass0:.6f}\n")

    # -----------------------------------------------------------------
    # Time loop
    # -----------------------------------------------------------------
    history = [{"phi": phi.copy(), "u": u.copy(), "v": v.copy(),
                "p": np.zeros_like(phi), "t": 0.0}]

    t    = 0.0
    step = 0
    n_steps  = int(math.ceil(t_end / dt))
    log_freq = max(1, n_steps // 10)   # print every ~10%

    while t < t_end - 1e-12:
        dt_actual = min(dt, t_end - t)
        u, v, phi, p = step_ns(u, v, phi, dt_actual, mesh, cfg, K2)
        t    += dt_actual
        step += 1

        if step % save_freq == 0 or t >= t_end - 1e-12:
            history.append({
                "phi": phi.copy(),
                "u":   u.copy(),
                "v":   v.copy(),
                "p":   p.copy(),
                "t":   t,
            })

        if step % log_freq == 0 or t >= t_end - 1e-12:
            mass      = float(phi.sum()) * mesh.dx * mesh.dy
            mass_err  = abs(mass - mass0)
            u_max     = max(float(np.abs(u).max()), float(np.abs(v).max()))
            print(f"step {step:5d}  t={t:.3f}  "
                  f"phi=[{phi.min():.3f},{phi.max():.3f}]  "
                  f"mass_err={mass_err:.2e}  |u|_max={u_max:.3f}")

    return history
