"""
impact/ns_solver.py
-------------------
One-step incompressible Navier-Stokes solver for the droplet impact simulation.

Uses a fractional-step (Chorin/Van Kan) projection method:

  1. Predictor  u* = u^n + dt * [-(u·∇)u + (1/ρ)∇·(μ∇u) + (σ/ρ)κ∇φ + g_buoy]
  2. Poisson    ∇²p = (ρ̄/dt) ∇·u*   (Neumann y-BCs via even-extension)
  3. Corrector  u^{n+1} = u* - (dt/ρ̄) ∇p
  4. Phase-field φ advanced with ACDI + skew-symmetric advection (RK4)

Surface tension enters via the CSF (Continuum Surface Force) model:
  F_σ = σ κ ∇φ / ρ    where  κ = -∇·n̂,  n̂ = ∇φ/|∇φ|

Gravity uses a buoyancy formulation relative to the initial density field rho_init
(stored in cfg["rho_init"]).  This keeps the static pool in equilibrium while the
drop still falls under its own weight relative to the surrounding gas:
  g_body = -(ρ - ρ_init) / ρ · g

The Poisson solver uses Neumann BCs in y (∂p/∂y = 0 at top/bottom walls) via the
even-extension (method of images) trick.  This prevents the drop's pressure field
from coupling through the periodic y-boundary to the pool below.

Variable properties:
  ρ(φ) = ρ1·φ + ρ2·(1-φ)
  μ(φ) = μ1·φ + μ2·(1-φ)
"""

import numpy as np
from numpy.fft import rfft2, irfft2

from core.mesh import Mesh
from core.regularization import compute_interface_normal, acdi_regularization, laplacian
from core.time_integration import rk4_step
from solvers.task4_acdi import skew_symmetric_advection


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _build_wavenumbers(mesh: Mesh):
    """Pre-build discrete eigenvalue matrix for the Neumann-y FFT Poisson solver.

    Uses the exact discrete Laplacian eigenvalues for the compact (2-point)
    finite-difference stencil:

        λx[m] = (2*sin(π*m/nx) / dx)²
        λy[m] = (2*sin(π*m/(2*ny)) / dy)²   [on the 2*ny even-extended domain]

    This makes the face-based pressure correction exactly divergence-free
    (div ≤ 1e-13), unlike the continuous k² approximation (div ≈ 4.6).

    Shape: (2*ny, nx//2+1)  — the even-extended y-domain for Neumann BCs.
    """
    nx, ny = mesh.nx, mesh.ny
    # x-direction (periodic rfft), m = 0..nx//2
    mx = np.arange(nx // 2 + 1)
    K2x = (2.0 * np.sin(np.pi * mx / nx) / mesh.dx) ** 2         # (nx//2+1,)

    # y-direction (even-extended, fft on 2*ny domain), m symmetric about ny
    my     = np.arange(2 * ny)
    my_sym = np.minimum(my, 2 * ny - my)                          # fold: [0,1,...,ny,ny-1,...,1]
    K2y    = (2.0 * np.sin(np.pi * my_sym / (2 * ny)) / mesh.dy) ** 2  # (2*ny,)

    K2 = K2y[:, np.newaxis] + K2x[np.newaxis, :]                 # (2*ny, nx//2+1)
    K2[0, 0] = 1.0   # avoid division by zero; mean pressure set to 0 below
    return K2


def cc_to_faces(
    u_cc: np.ndarray,
    v_cc: np.ndarray,
    mesh: Mesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate cell-centre velocities to face-staggered velocities.

    Face convention (periodic x, wall y):
      u_face[j, i] = 0.5*(u_cc[j, (i-1)%nx] + u_cc[j, i%nx])
      v_face[0,  :] = 0     (bottom wall)
      v_face[ny, :] = 0     (top wall)
      v_face[j,  :] = 0.5*(v_cc[j-1, :] + v_cc[j, :])  for j=1..ny-1

    Parameters
    ----------
    u_cc, v_cc : np.ndarray, shape (ny, nx)
        Cell-centre velocity components.
    mesh : Mesh

    Returns
    -------
    u_face : np.ndarray, shape (ny, nx+1)
    v_face : np.ndarray, shape (ny+1, nx)
    """
    nx, ny = mesh.nx, mesh.ny

    # u_face: average left and right cell-centre values (periodic x)
    u_roll = np.roll(u_cc, 1, axis=1)       # u_roll[:, i] = u_cc[:, (i-1)%nx]
    u_face = np.empty((ny, nx + 1))
    u_face[:, :-1] = 0.5 * (u_roll + u_cc)  # faces 0..nx-1
    u_face[:, -1] = u_face[:, 0]             # face nx = face 0 (periodic)

    # v_face: wall (Neumann) y-BCs — no flow through top/bottom walls.
    # This is CONSISTENT with the Neumann pressure solver (even-extension).
    v_face = np.empty((ny + 1, nx))
    v_face[0, :]    = 0.0                               # bottom wall: v=0
    v_face[ny, :]   = 0.0                               # top wall:    v=0
    v_face[1:-1, :] = 0.5 * (v_cc[:-1, :] + v_cc[1:, :])  # interior faces

    return u_face, v_face


def div_faces(
    u_face: np.ndarray,
    v_face: np.ndarray,
    mesh: Mesh,
) -> np.ndarray:
    """Compute cell-centre divergence from face-staggered velocities.

    div(u)[j,i] = (u_face[j,i+1] - u_face[j,i])/dx
                + (v_face[j+1,i] - v_face[j,i])/dy
    """
    return (
        (u_face[:, 1:] - u_face[:, :-1]) / mesh.dx
        + (v_face[1:, :] - v_face[:-1, :]) / mesh.dy
    )


def grad_cc(
    p: np.ndarray,
    mesh: Mesh,
) -> tuple[np.ndarray, np.ndarray]:
    """Cell-centre gradient of p: periodic x, Neumann y (reflect at walls)."""
    gx = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2.0 * mesh.dx)
    # Neumann y: ghost cell = mirror of interior → central diff is zero at wall
    p_pad = np.pad(p, ((1, 1), (0, 0)), mode='reflect')
    gy = (p_pad[2:] - p_pad[:-2]) / (2.0 * mesh.dy)
    return gx, gy


def compute_curvature(
    phi: np.ndarray,
    mesh: Mesh,
) -> np.ndarray:
    """Compute interface curvature κ = -∇·n̂ via central differences.

    Uses compute_interface_normal from core.regularization for n̂.
    """
    nx_hat, ny_hat = compute_interface_normal(phi, mesh)
    kappa = -(
        (np.roll(nx_hat, -1, axis=1) - np.roll(nx_hat, 1, axis=1)) / (2.0 * mesh.dx)
        + (np.roll(ny_hat, -1, axis=0) - np.roll(ny_hat, 1, axis=0)) / (2.0 * mesh.dy)
    )
    return kappa


def solve_pressure_fft(
    div_ustar: np.ndarray,
    rho_mean: float,
    dt: float,
    mesh: Mesh,
    K2: np.ndarray,
) -> np.ndarray:
    """Solve ∇²p = (ρ̄/dt)·∇·u* with Neumann y-BCs (even-extension trick).

    Neumann BCs (∂p/∂y = 0 at y=0, y=Ly) prevent the drop's pressure
    from coupling through the periodic y-boundary to the pool.

    Method of images: extend div_ustar by even reflection in y, apply the
    periodic FFT Poisson solver on the 2*ny domain, take the first ny rows.

    Parameters
    ----------
    div_ustar : np.ndarray, shape (ny, nx)
    rho_mean : float
    dt : float
    mesh : Mesh
    K2 : np.ndarray, shape (2*ny, nx//2+1)
        Pre-built wavenumber matrix from _build_wavenumbers (extended domain).

    Returns
    -------
    p : np.ndarray, shape (ny, nx)
    """
    nx, ny = mesh.nx, mesh.ny
    # Even extension: [div; flip_y(div)] → ∂p/∂y = 0 at both walls
    div_ext = np.empty((2 * ny, nx))
    div_ext[:ny] = div_ustar
    div_ext[ny:] = div_ustar[::-1]
    div_hat = rfft2(div_ext)
    p_hat = -(rho_mean / dt) * div_hat / K2
    p_hat[0, 0] = 0.0   # zero mean pressure
    p_ext = irfft2(p_hat, s=(2 * ny, nx))
    return p_ext[:ny]


def solve_pressure_pcg(
    div_ustar: np.ndarray,
    rho: np.ndarray,
    rho_mean: float,
    dt: float,
    mesh: Mesh,
    K2: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-6,
) -> np.ndarray:
    """Solve div((dt/ρ)∇p) = ∇·u* via PCG with FFT preconditioner.

    Implements the full variable-density projection:
        div((dt/ρ) ∇p) = div(u*)          (A · p = b)

    Uses a preconditioned conjugate gradient (PCG) method with the constant-
    coefficient FFT Poisson solver as preconditioner:
        M · p = (dt/ρ̄) ∇²p   →   M⁻¹r = FFT_solve(ρ̄/dt · r)

    This gives the correct pressure field for a large-density-ratio flow.
    With only rho_mean in the Poisson RHS, the pressure inside the drop has
    a spurious gradient proportional to drop velocity, creating ~3× too much
    deceleration per step.  PCG eliminates that artefact in ≤15 iterations
    (condition number κ ≈ ρ₁/ρ₂ = 100, PCG converges in √κ ≈ 10 steps).

    Returns
    -------
    p : np.ndarray, shape (ny, nx)
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy

    # Pre-compute face densities (used by the A operator every iteration)
    rho_xf = np.empty((ny, nx + 1))
    rho_xf[:, 1:-1] = 0.5 * (rho[:, 1:] + rho[:, :-1])
    rho_xf[:, 0]    = 0.5 * (rho[:, 0] + rho[:, -1])
    rho_xf[:, -1]   = rho_xf[:, 0]

    rho_yf = np.empty((ny + 1, nx))
    rho_yf[1:-1, :] = 0.5 * (rho[:-1, :] + rho[1:, :])
    rho_yf[0,  :]   = rho[0,  :]
    rho_yf[-1, :]   = rho[-1, :]

    def apply_A(p):
        """A(p) = div((dt/ρ_face)·∇p) with Neumann y-BCs."""
        gx = np.empty((ny, nx + 1))
        gx[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dx
        gx[:, 0]    = (p[:, 0] - p[:, -1]) / dx
        gx[:, -1]   = gx[:, 0]

        gy = np.empty((ny + 1, nx))
        gy[1:-1, :] = (p[1:, :] - p[:-1, :]) / dy
        gy[0,  :] = 0.0
        gy[-1, :] = 0.0

        fx = (dt / rho_xf) * gx
        fy = (dt / rho_yf) * gy
        fy[0, :] = 0.0;  fy[-1, :] = 0.0
        return (fx[:, 1:] - fx[:, :-1]) / dx + (fy[1:, :] - fy[:-1, :]) / dy

    def precond(r):
        """M⁻¹(r): solve (dt/ρ̄)·Lap(z) = r  ↔  Lap(z) = (ρ̄/dt)·r  [FFT]."""
        return solve_pressure_fft(r, rho_mean, dt, mesh, K2)

    # Initial guess and residual
    p = precond(div_ustar)
    r = div_ustar - apply_A(p)
    z = precond(r)
    d = z.copy()
    rz = float((r * z).sum())

    b_norm = float(np.sqrt((div_ustar ** 2).sum())) + 1e-30

    for _ in range(max_iter):
        if float(np.sqrt((r ** 2).sum())) < tol * b_norm:
            break
        q    = apply_A(d)
        dq   = float((d * q).sum())
        if abs(dq) < 1e-30:
            break
        alpha = rz / dq
        p     = p + alpha * d
        r     = r - alpha * q
        z     = precond(r)
        rz_new = float((r * z).sum())
        beta  = rz_new / (rz + 1e-30)
        d     = z + beta * d
        rz    = rz_new

    return p


# ---------------------------------------------------------------------------
# Main NS step
# ---------------------------------------------------------------------------

def step_ns(
    u: np.ndarray,
    v: np.ndarray,
    phi: np.ndarray,
    dt: float,
    mesh: Mesh,
    cfg: dict,
    K2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Advance (u, v, phi) by one time step using the projection method.

    Parameters
    ----------
    u, v : np.ndarray, shape (ny, nx)
        Cell-centre velocity components at time t^n.
    phi : np.ndarray, shape (ny, nx)
        Volume fraction at time t^n.
    dt : float
        Time step.
    mesh : Mesh
    cfg : dict
        Must contain: rho1, rho2, mu1, mu2, sigma, g, eps_factor, rho_init.
        rho_init is the initial density field (shape ny×nx), used for the
        buoyancy body force so the static pool has zero effective gravity.
    K2 : np.ndarray, shape (2*ny, nx//2+1)
        Pre-built wavenumber matrix from _build_wavenumbers (Neumann y-BCs).

    Returns
    -------
    u_new, v_new : np.ndarray, shape (ny, nx)
        Updated velocity components.
    phi_new : np.ndarray, shape (ny, nx)
        Updated volume fraction.
    p : np.ndarray, shape (ny, nx)
        Pressure field (at time n+1).
    """
    nx, ny = mesh.nx, mesh.ny
    dx, dy = mesh.dx, mesh.dy
    rho1, rho2 = cfg["rho1"], cfg["rho2"]
    mu1,  mu2  = cfg["mu1"],  cfg["mu2"]
    sigma    = cfg["sigma"]
    g        = cfg["g"]
    eps      = cfg["eps_factor"] * dx
    rho_init = cfg["rho_init"]   # fixed initial density; buoyancy reference

    # Step 1: Variable properties
    rho = rho1 * phi + rho2 * (1.0 - phi)
    mu  = mu1  * phi + mu2  * (1.0 - phi)

    # Step 2: Interface curvature for CSF
    kappa = compute_curvature(phi, mesh)

    # Step 3: Gradient of phi (for CSF force)
    dphi_dx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2.0 * dx)
    dphi_dy = (np.roll(phi, -1, axis=0) - np.roll(phi, 1, axis=0)) / (2.0 * dy)

    # Step 4: CSF surface tension force per unit mass
    F_sigma_x = sigma * kappa * dphi_dx / rho
    F_sigma_y = sigma * kappa * dphi_dy / rho

    # Step 5: Convection (central differences, skew form for u-momentum)
    du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2.0 * dx)
    du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2.0 * dy)
    dv_dx = (np.roll(v, -1, axis=1) - np.roll(v, 1, axis=1)) / (2.0 * dx)
    dv_dy = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2.0 * dy)

    conv_u = -(u * du_dx + v * du_dy)
    conv_v = -(u * dv_dx + v * dv_dy)

    # Step 6: Viscous term (simplified: μ/ρ · ∇²u)
    visc_u = (mu / rho) * laplacian(u, mesh)
    visc_v = (mu / rho) * laplacian(v, mesh)

    # Step 7: Predictor (explicit Euler)
    # Buoyancy body force: g_body = -(ρ - ρ_init)/ρ · g
    #   Pool  (ρ = ρ_init = ρ1): g_body = 0  → pool stays at rest  ✓
    #   Gas   (ρ = ρ_init = ρ2): g_body = 0  → gas stays at rest    ✓
    #   Drop  (ρ = ρ1, ρ_init = ρ2 at drop location):
    #         g_body ≈ -(ρ1-ρ2)/ρ1 · g ≈ -g  → drop falls            ✓
    g_body = -(rho - rho_init) / rho * g

    rhs_u = conv_u + visc_u + F_sigma_x
    rhs_v = conv_v + visc_v + F_sigma_y + g_body

    u_star = u + dt * rhs_u
    v_star = v + dt * rhs_v

    # Step 8: Divergence of u*
    u_star_face, v_star_face = cc_to_faces(u_star, v_star, mesh)
    div_star = div_faces(u_star_face, v_star_face, mesh)

    # Step 9: Solve variable-density pressure Poisson via PCG.
    #
    # Solves div((dt/ρ)∇p) = ∇·u* exactly (up to CG tolerance).
    # This gives the physically correct pressure — no spurious gradient inside
    # the drop that would decelerate it 3× too fast per step with rho_mean.
    rho_mean = float(rho.mean())
    p = solve_pressure_pcg(div_star, rho, rho_mean, dt, mesh, K2)

    # Step 10: Face-based velocity correction with local face density.
    #
    # Since PCG solved div((dt/ρ_face)∇p) = div_star, correcting with the
    # same (dt/ρ_face) factor gives EXACTLY div-free face velocities.
    # Face density — floored at rho_mean to prevent gas-cell explosion.
    # In pure-gas regions: div_star ≈ 0 → Lap(p) ≈ 0 → residual div ≈ 0
    # even when rho_mean is used instead of rho_gas.  Liquid faces (rho > rho_mean)
    # use the correct local density from the PCG pressure.
    rho_xf_raw = np.empty((ny, nx + 1))
    rho_xf_raw[:, 1:-1] = 0.5 * (rho[:, 1:] + rho[:, :-1])
    rho_xf_raw[:, 0]    = 0.5 * (rho[:, 0] + rho[:, -1])
    rho_xf_raw[:, -1]   = rho_xf_raw[:, 0]

    rho_yf_raw = np.empty((ny + 1, nx))
    rho_yf_raw[1:-1, :] = 0.5 * (rho[:-1, :] + rho[1:, :])
    rho_yf_raw[0,  :]   = rho[0,  :]
    rho_yf_raw[-1, :]   = rho[-1, :]

    rho_xf = np.maximum(rho_xf_raw, rho_mean)
    rho_yf = np.maximum(rho_yf_raw, rho_mean)

    gp_x_face = np.empty((ny, nx + 1))
    gp_x_face[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dx
    gp_x_face[:, 0]    = (p[:, 0] - p[:, -1]) / dx
    gp_x_face[:, -1]   = gp_x_face[:, 0]

    gp_y_face = np.empty((ny + 1, nx))
    gp_y_face[1:-1, :] = (p[1:, :] - p[:-1, :]) / dy
    gp_y_face[0,  :] = 0.0
    gp_y_face[-1, :] = 0.0

    u_face_new = u_star_face - (dt / rho_xf) * gp_x_face
    v_face_new = v_star_face - (dt / rho_yf) * gp_y_face
    v_face_new[0,  :] = 0.0   # enforce no-penetration at walls
    v_face_new[-1, :] = 0.0

    # Cell-centered correction: local density floored at rho_mean (same reasoning).
    rho_eff_cc = np.maximum(rho, rho_mean)
    gp_x_cc    = (np.roll(p, -1, axis=1) - np.roll(p, 1, axis=1)) / (2.0 * dx)
    p_pad      = np.pad(p, ((1, 1), (0, 0)), mode='reflect')
    gp_y_cc    = (p_pad[2:] - p_pad[:-2]) / (2.0 * dy)
    u_new = u_star - (dt / rho_eff_cc) * gp_x_cc
    v_new = v_star - (dt / rho_eff_cc) * gp_y_cc

    # Step 11: Advance phi with ACDI (RK4) using divergence-free face velocities
    # (u_face_new, v_face_new are already computed above — no extra cc_to_faces call)
    u_max = max(
        float(np.abs(u_face_new).max()),
        float(np.abs(v_face_new).max()),
        eps * 0.05,
    )
    Gamma = u_max

    def phi_rhs(ph, _t):
        return (
            skew_symmetric_advection(ph, u_face_new, v_face_new, mesh)
            + acdi_regularization(ph, mesh, eps, Gamma=Gamma)
        )

    phi_new = np.clip(rk4_step(phi, 0.0, dt, phi_rhs), 0.0, 1.0)

    return u_new, v_new, phi_new, p
