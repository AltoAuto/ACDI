"""
postprocessing/plotting.py
--------------------------
ME 5351 HW2 - Visualisation routines for the phase-field solver.

All functions use matplotlib as the backend.  Figures are saved to disk
as PNG/PDF in the task-specific results/ directory.

Functions
---------
plot_phi_field
    Filled colour map of phi(x, y) at a given time, with the phi = 0.5
    iso-contour drawn as a solid white/black line to mark the interface.

plot_interface_contour
    Overlay the phi = 0.5 contour from multiple tasks on a single axes for
    direct visual comparison (useful for the shear flow test at t = T).

plot_convergence
    Log-log plot of L1/L2/Linf error vs. grid spacing (mesh refinement
    study) with reference slope lines for 1st and 2nd order.

plot_mass_history
    Time series of total mass  M(t) = integral(phi) * dx * dy  showing
    conservation properties of each task's solver.

animate_phi
    Create an MP4 / GIF animation of the phi field evolving in time using
    matplotlib FuncAnimation.  Saved to the output directory.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional

from core.mesh import Mesh

# Portfolio-quality defaults
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_phi_field(
    phi: np.ndarray,
    mesh: Mesh,
    t: float,
    title: str = "",
    cmap: str = "RdBu_r",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot a filled colour map of phi with the phi = 0.5 interface contour.

    Parameters
    ----------
    phi : np.ndarray, shape (ny, nx)
        Volume-fraction field.
    mesh : Mesh
        Computational mesh (provides XC, YC).
    t : float
        Current simulation time (shown in title).
    title : str
        Additional title string.
    cmap : str
        Matplotlib colormap name.
    save_path : str or None
        If provided, save the figure to this path.
    show : bool
        If True, call plt.show().

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    pcm = ax.pcolormesh(mesh.XC, mesh.YC, phi, cmap=cmap, vmin=0, vmax=1,
                        shading="auto")
    ax.contour(mesh.XC, mesh.YC, phi, levels=[0.5], colors="k", linewidths=1.2)
    fig.colorbar(pcm, ax=ax, label=r"$\phi$", shrink=0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"$\\phi(x,y)$ at $t = {t:.4f}$")

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_interface_contour(
    phi_dict: dict[str, np.ndarray],
    mesh: Mesh,
    t: float,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Overlay phi = 0.5 contours from multiple tasks on one figure.

    Parameters
    ----------
    phi_dict : dict
        Mapping from label (e.g. 'Task 1', 'Task 4') to phi array.
    mesh : Mesh
        Shared computational mesh.
    t : float
        Simulation time shown in title.
    title : str
        Additional title.
    save_path : str or None
    show : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (label, phi) in enumerate(phi_dict.items()):
        color = colors[idx % len(colors)]
        ax.contour(mesh.XC, mesh.YC, phi, levels=[0.5], colors=[color],
                   linewidths=1.5)
        # Invisible line for legend
        ax.plot([], [], color=color, linewidth=1.5, label=label)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Interface comparison at $t = {t:.4f}$")

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_convergence(
    dx_list: list[float],
    error_dict: dict[str, list[float]],
    title: str = "Convergence study",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Log-log convergence plot (error vs. mesh spacing).

    Parameters
    ----------
    dx_list : list of float
        Mesh spacings used (x-axis).
    error_dict : dict
        Mapping from label (e.g. 'L1 Task 1') to list of errors.
    title : str
    save_path : str or None
    show : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    dx_arr = np.array(dx_list)

    for label, errors in error_dict.items():
        ax.loglog(dx_arr, errors, "o-", label=label, markersize=5)

    # Reference slopes
    ref_x = np.array([dx_arr[0], dx_arr[-1]])
    ax.loglog(ref_x, errors[0] * (ref_x / ref_x[0])**1,
              "k--", alpha=0.4, label="$O(\\Delta x)$")
    ax.loglog(ref_x, errors[0] * (ref_x / ref_x[0])**2,
              "k:", alpha=0.4, label="$O(\\Delta x^2)$")

    ax.set_xlabel(r"$\Delta x$")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def plot_mass_history(
    t_history: list[float],
    mass_history: list[float],
    label: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Plot total mass M(t) vs. time.

    Parameters
    ----------
    t_history : list of float
        Simulation times.
    mass_history : list of float
        Total mass at each saved time step.
    label : str
        Legend label.
    save_path : str or None
    show : bool

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(t_history, mass_history, "-", label=label if label else None)
    ax.set_xlabel("Time $t$")
    ax.set_ylabel("Total mass $M(t)$")
    ax.set_title("Mass conservation")
    if label:
        ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)
    return fig


def animate_phi(
    phi_history: list[np.ndarray],
    t_history: list[float],
    mesh: Mesh,
    save_path: str,
    fps: int = 10,
    cmap: str = "RdBu_r",
) -> FuncAnimation:
    """Create and save an animation of the phi field evolving in time.

    Parameters
    ----------
    phi_history : list of np.ndarray
        Sequence of phi snapshots.
    t_history : list of float
        Corresponding simulation times.
    mesh : Mesh
        Computational mesh.
    save_path : str
        Output file path (.mp4 or .gif).
    fps : int
        Frames per second.
    cmap : str
        Colormap.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
    pcm = ax.pcolormesh(mesh.XC, mesh.YC, phi_history[0], cmap=cmap,
                        vmin=0, vmax=1, shading="auto")
    contour_coll = [ax.contour(mesh.XC, mesh.YC, phi_history[0],
                               levels=[0.5], colors="k", linewidths=1)]
    fig.colorbar(pcm, ax=ax, label=r"$\phi$", shrink=0.85)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    title = ax.set_title(f"$t = {t_history[0]:.4f}$")

    def update(frame):
        pcm.set_array(phi_history[frame].ravel())

        # Remove old contours in a way that is compatible across matplotlib versions.
        old_contour = contour_coll[0]
        old_collections = getattr(old_contour, "collections", None)
        if old_collections is not None:
            for c in old_collections:
                c.remove()
        else:
            old_contour.remove()
        contour_coll[0] = ax.contour(mesh.XC, mesh.YC, phi_history[frame],
                                     levels=[0.5], colors="k", linewidths=1)
        title.set_text(f"$t = {t_history[frame]:.4f}$")
        return [pcm]

    anim = FuncAnimation(fig, update, frames=len(phi_history), interval=1000 // fps)
    anim.save(save_path, fps=fps, writer="pillow" if save_path.endswith(".gif") else "ffmpeg")
    plt.close(fig)
    return anim
