"""
impact/plotting.py
------------------
4-panel dark-themed animation for the droplet impact simulation.

Panels:
  [0,0] Volume fraction phi  (RdBu_r, 0-1) + white phi=0.5 contour
  [0,1] Velocity magnitude   |u| (hot colormap, 0 to 2*U_init)
  [1,0] Pressure p           (RdBu diverging, symmetric ±p_max)
  [1,1] Vorticity ω=∂v/∂x-∂u/∂y  (bwr, symmetric ±ω_max)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize, TwoSlopeNorm

from core.mesh import Mesh


def _vorticity(u: np.ndarray, v: np.ndarray, mesh: Mesh) -> np.ndarray:
    """ω = ∂v/∂x − ∂u/∂y  (central differences, periodic)."""
    dvdx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * mesh.dx)
    dudy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * mesh.dy)
    return dvdx - dudy


def animate_impact(
    history: list[dict],
    mesh: Mesh,
    save_path: str,
    fps: int = 15,
    drop_U: float = 1.0,
) -> None:
    """Create and save a 4-panel animation of the impact simulation.

    Parameters
    ----------
    history : list of dict
        Snapshots from run_impact. Each dict has: phi, u, v, p, t.
    mesh : Mesh
        Computational mesh (for axis extents and derivatives).
    save_path : str
        Output file path (e.g. 'impact.gif').
    fps : int
        Frames per second.
    drop_U : float
        Initial drop velocity (used to set velocity magnitude colormap limit).
    """
    if not history:
        print("No snapshots to animate.")
        return

    print(f"Generating animation: {len(history)} frames -> {save_path}")

    extent = [0, mesh.Lx, 0, mesh.Ly]

    # Pre-compute global limits for stable colormaps
    all_p   = [d["p"]   for d in history]
    all_u   = [d["u"]   for d in history]
    all_v   = [d["v"]   for d in history]
    all_phi = [d["phi"] for d in history]

    p_max  = max(np.abs(p).max() for p in all_p) or 1.0
    u_max  = 2.0 * drop_U   # cap at 2× initial drop velocity
    om_max = max(
        abs(_vorticity(u, v, mesh)).max()
        for u, v in zip(all_u, all_v)
    ) or 1.0

    # --- Figure setup (dark theme) ---
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), facecolor="black")
    fig.subplots_adjust(left=0.07, right=0.93, top=0.92, bottom=0.06,
                        hspace=0.30, wspace=0.35)

    ax_phi = axes[0, 0]
    ax_vel = axes[0, 1]
    ax_p   = axes[1, 0]
    ax_vrt = axes[1, 1]

    # Initialise with first frame
    frame0 = history[0]
    phi0   = frame0["phi"]
    u0, v0 = frame0["u"], frame0["v"]
    p0     = frame0["p"]
    om0    = _vorticity(u0, v0, mesh)
    umag0  = np.sqrt(u0 ** 2 + v0 ** 2)

    im_phi = ax_phi.imshow(phi0, origin="lower", extent=extent,
                           cmap="RdBu_r", vmin=0, vmax=1, aspect="equal")
    ct_phi = ax_phi.contour(mesh.XC, mesh.YC, phi0, levels=[0.5],
                            colors="white", linewidths=0.8)
    ax_phi.set_title("Volume Fraction φ", color="white", fontsize=11)
    ax_phi.set_xlabel("x", color="white"); ax_phi.set_ylabel("y", color="white")
    fig.colorbar(im_phi, ax=ax_phi, fraction=0.046, pad=0.04)

    im_vel = ax_vel.imshow(umag0, origin="lower", extent=extent,
                           cmap="hot", vmin=0, vmax=u_max, aspect="equal")
    ax_vel.set_title("|u| Velocity Magnitude", color="white", fontsize=11)
    ax_vel.set_xlabel("x", color="white"); ax_vel.set_ylabel("y", color="white")
    fig.colorbar(im_vel, ax=ax_vel, fraction=0.046, pad=0.04)

    im_p = ax_p.imshow(p0, origin="lower", extent=extent,
                       cmap="RdBu", vmin=-p_max, vmax=p_max, aspect="equal")
    ax_p.set_title("Pressure p", color="white", fontsize=11)
    ax_p.set_xlabel("x", color="white"); ax_p.set_ylabel("y", color="white")
    fig.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)

    im_vrt = ax_vrt.imshow(om0, origin="lower", extent=extent,
                           cmap="bwr", vmin=-om_max, vmax=om_max, aspect="equal")
    ax_vrt.set_title("Vorticity ω = ∂v/∂x − ∂u/∂y", color="white", fontsize=11)
    ax_vrt.set_xlabel("x", color="white"); ax_vrt.set_ylabel("y", color="white")
    fig.colorbar(im_vrt, ax=ax_vrt, fraction=0.046, pad=0.04)

    suptitle = fig.suptitle("Droplet Impact  t=0.000", color="white",
                             fontsize=13, y=0.96)

    # Store contour artists for removal on each frame update
    contour_store = [ct_phi]

    def update(frame_idx):
        data = history[frame_idx]
        phi  = data["phi"]
        u    = data["u"]
        v    = data["v"]
        p    = data["p"]
        t    = data["t"]
        om   = _vorticity(u, v, mesh)
        umag = np.sqrt(u ** 2 + v ** 2)

        im_phi.set_data(phi)
        im_vel.set_data(umag)
        im_p.set_data(p)
        im_vrt.set_data(om)
        suptitle.set_text(f"Droplet Impact  t={t:.3f}")

        # Remove old contour and draw new one (compatible with matplotlib ≥3.8)
        try:
            contour_store[0].remove()
        except AttributeError:
            for coll in contour_store[0].collections:
                coll.remove()
        ct = ax_phi.contour(mesh.XC, mesh.YC, phi, levels=[0.5],
                            colors="white", linewidths=0.8)
        contour_store[0] = ct

        return [im_phi, im_vel, im_p, im_vrt, suptitle]

    ani = animation.FuncAnimation(
        fig, update, frames=len(history),
        interval=1000 // fps, blit=False,
    )

    ani.save(save_path, writer="pillow", fps=fps)
    plt.close(fig)
    print(f"Saved: {save_path}")
