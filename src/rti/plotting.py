"""
rti/plotting.py
---------------
Side-by-side animation of CDI vs ACDI RTI simulations.

Layout (2 rows x 2 cols):
  Top row    : phi fields  (RdBu_r colormap, phi=0.5 contour)
  Bottom row : |grad phi|  (hot colormap — sharper interface = brighter)

The gradient row makes the CDI vs ACDI difference immediately visible:
ACDI maintains a thin bright line (high |grad phi|); CDI spreads into a
dimmer wider band as the interface diffuses over time.

Output: GIF (via Pillow) or MP4 (via ffmpeg if available).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from core.mesh import Mesh


def _grad_mag(phi: np.ndarray, mesh: Mesh) -> np.ndarray:
    """Central-difference gradient magnitude |grad phi|."""
    gx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * mesh.dx)
    gy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * mesh.dy)
    return np.sqrt(gx**2 + gy**2)


def _interface_cells(phi: np.ndarray) -> int:
    """Count cells in the diffuse zone 0.05 < phi < 0.95."""
    return int(np.sum((phi > 0.05) & (phi < 0.95)))


def animate_comparison(
    cdi_hist: list[np.ndarray],
    acdi_hist: list[np.ndarray],
    t_hist: list[float],
    mesh: Mesh,
    save_path: str,
    fps: int = 15,
) -> None:
    """Create and save a 4-panel CDI vs ACDI animation.

    Top row : phi (phase field) for CDI and ACDI.
    Bottom  : |grad phi| (interface sharpness) for CDI and ACDI.

    ACDI's bottom panels stay bright and thin (sharp interface);
    CDI's bottom panels grow dimmer and wider as the interface diffuses.

    Parameters
    ----------
    cdi_hist, acdi_hist : list of np.ndarray, shape (ny, nx)
    t_hist              : list of float
    mesh                : Mesh
    save_path           : str  (.gif or .mp4)
    fps                 : int
    """
    n_frames = min(len(cdi_hist), len(acdi_hist), len(t_hist))
    extent = [0, mesh.Lx, 0, mesh.Ly]
    eps_ref = 1.5 * mesh.dx          # reference eps for grad_max
    grad_max = 1.0 / (4.0 * eps_ref) # max |grad phi| for perfect tanh profile
    XC, YC = mesh.XC, mesh.YC

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 10),
        gridspec_kw={"hspace": 0.35, "wspace": 0.15},
    )
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.tick_params(colors="0.7")
        ax.xaxis.label.set_color("0.7")
        ax.yaxis.label.set_color("0.7")
        for spine in ax.spines.values():
            spine.set_edgecolor("0.4")

    # --- Row 0: phi fields ---
    im_cdi = axes[0, 0].imshow(
        cdi_hist[0], origin="lower", extent=extent,
        cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="equal",
    )
    im_acdi = axes[0, 1].imshow(
        acdi_hist[0], origin="lower", extent=extent,
        cmap="RdBu_r", vmin=0.0, vmax=1.0, aspect="equal",
    )
    axes[0, 0].set_title("CDI  —  phi field", color="white", fontsize=12)
    axes[0, 1].set_title("ACDI  —  phi field", color="white", fontsize=12)

    cont_cdi  = [axes[0, 0].contour(XC, YC, cdi_hist[0],
                                    levels=[0.5], colors="white", linewidths=0.8)]
    cont_acdi = [axes[0, 1].contour(XC, YC, acdi_hist[0],
                                    levels=[0.5], colors="white", linewidths=0.8)]

    fig.colorbar(im_cdi,  ax=axes[0, 0], fraction=0.046, pad=0.04,
                 label="phi").ax.yaxis.label.set_color("0.7")
    fig.colorbar(im_acdi, ax=axes[0, 1], fraction=0.046, pad=0.04,
                 label="phi").ax.yaxis.label.set_color("0.7")

    # --- Row 1: |grad phi| (sharpness) ---
    g0_cdi  = _grad_mag(cdi_hist[0],  mesh)
    g0_acdi = _grad_mag(acdi_hist[0], mesh)
    im_gcdi = axes[1, 0].imshow(
        g0_cdi, origin="lower", extent=extent,
        cmap="hot", vmin=0.0, vmax=grad_max, aspect="equal",
    )
    im_gacdi = axes[1, 1].imshow(
        g0_acdi, origin="lower", extent=extent,
        cmap="hot", vmin=0.0, vmax=grad_max, aspect="equal",
    )
    sub_cdi  = axes[1, 0].set_title(
        f"CDI  |grad phi|   diffuse cells: {_interface_cells(cdi_hist[0])}",
        color="white", fontsize=11,
    )
    sub_acdi = axes[1, 1].set_title(
        f"ACDI |grad phi|   diffuse cells: {_interface_cells(acdi_hist[0])}",
        color="white", fontsize=11,
    )
    fig.colorbar(im_gcdi,  ax=axes[1, 0], fraction=0.046, pad=0.04,
                 label="|grad phi|").ax.yaxis.label.set_color("0.7")
    fig.colorbar(im_gacdi, ax=axes[1, 1], fraction=0.046, pad=0.04,
                 label="|grad phi|").ax.yaxis.label.set_color("0.7")

    for ax in axes.flat:
        ax.set_xlabel("x", fontsize=9)
        ax.set_ylabel("y", fontsize=9)

    title = fig.suptitle(
        f"Rayleigh-Taylor Instability   t = {t_hist[0]:.4f}",
        color="white", fontsize=14, y=0.98,
    )

    def _remove_contour(c):
        try:
            c.remove()
        except AttributeError:
            for coll in c.collections:
                coll.remove()

    def update(frame):
        phi_c = cdi_hist[frame]
        phi_a = acdi_hist[frame]

        im_cdi.set_data(phi_c)
        im_acdi.set_data(phi_a)

        _remove_contour(cont_cdi[0])
        _remove_contour(cont_acdi[0])
        cont_cdi[0]  = axes[0, 0].contour(XC, YC, phi_c,
                                          levels=[0.5], colors="white", linewidths=0.8)
        cont_acdi[0] = axes[0, 1].contour(XC, YC, phi_a,
                                          levels=[0.5], colors="white", linewidths=0.8)

        gc = _grad_mag(phi_c, mesh)
        ga = _grad_mag(phi_a, mesh)
        im_gcdi.set_data(gc)
        im_gacdi.set_data(ga)

        nc = _interface_cells(phi_c)
        na = _interface_cells(phi_a)
        sub_cdi.set_text(f"CDI  |grad phi|   diffuse cells: {nc}")
        sub_acdi.set_text(f"ACDI |grad phi|   diffuse cells: {na}")

        title.set_text(
            f"Rayleigh-Taylor Instability   t = {t_hist[frame]:.4f}"
        )
        return [im_cdi, im_acdi, im_gcdi, im_gacdi]

    anim = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False,
    )

    saved_path = save_path
    if save_path.endswith(".gif"):
        writer = animation.PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
    else:
        if animation.FFMpegWriter.isAvailable():
            anim.save(save_path, writer="ffmpeg", fps=fps)
        else:
            gif_path = save_path.rsplit(".", 1)[0] + ".gif"
            print(f"ffmpeg unavailable; saving as GIF: {gif_path}")
            writer = animation.PillowWriter(fps=fps)
            anim.save(gif_path, writer=writer)
            saved_path = gif_path

    print(f"Animation saved: {saved_path}")
    plt.close(fig)
