"""
core/velocity_fields.py
-----------------------
ME 5351 HW2 - Prescribed divergence-free velocity fields.

Two velocity fields are required by the homework:

1. uniform_velocity
   Constant advection u = U0, v = 0 (Test Case 1).
   The default value is U0 = 5 (dimensionless units, problem statement).

2. shear_flow_velocity
   Time-periodic oscillating shear flow (Test Case 2), defined on the
   unit square [0, 1]^2 following LeVeque (1996) / Rider & Kothe (1998):

       u(x, y, t) =  sin^2(pi*x) * sin(2*pi*y) * cos(pi*t / T)
       v(x, y, t) = -sin(2*pi*x) * sin^2(pi*y) * cos(pi*t / T)

   where T is the period.  The cosine factor reverses the flow at t = T/2
   so that the drop returns to its original shape at t = T.

Both functions return face-normal velocities on the mesh faces, which is the
standard input format for finite-volume flux assembly.

Reference
---------
  Jain 2022, Section 4.2 (oscillating shear flow).
  LeVeque, R. J. (1996). J. Comput. Phys., 123, 187-192.
"""

import numpy as np
from .mesh import Mesh


def uniform_velocity(
    mesh: Mesh,
    U0: float = 5.0,
    V0: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return face-normal velocities for a uniform advection field.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh.
    U0 : float
        Uniform x-velocity component (default 5.0 per problem statement).
    V0 : float
        Uniform y-velocity component (default 0.0).

    Returns
    -------
    u_face : np.ndarray, shape (ny, nx+1)
        x-component velocity at east/west faces (i.e., on x-faces).
    v_face : np.ndarray, shape (ny+1, nx)
        y-component velocity at north/south faces (i.e., on y-faces).
    """
    u_face = U0 * np.ones((mesh.ny, mesh.nx + 1))
    v_face = V0 * np.ones((mesh.ny + 1, mesh.nx))
    return u_face, v_face


def shear_flow_velocity(
    mesh: Mesh,
    t: float,
    T: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return face-normal velocities for the oscillating shear flow.

    The stream function is:
        psi(x, y, t) = (1/pi) * sin^2(pi*x) * sin^2(pi*y) * cos(pi*t / T)

    which yields the divergence-free velocity components listed in the module
    docstring.  Velocities are interpolated to face centres.

    Parameters
    ----------
    mesh : Mesh
        The computational mesh (assumed unit square domain).
    t : float
        Current simulation time.
    T : float
        Period of the oscillating flow (default 2.0).

    Returns
    -------
    u_face : np.ndarray, shape (ny, nx+1)
        x-velocity at east/west faces.
    v_face : np.ndarray, shape (ny+1, nx)
        y-velocity at north/south faces.
    """
    cos_t = np.cos(np.pi * t / T)

    # x-faces: located at x = x0 + i*dx, y = yc[j]  (cell-centre y)
    xf = mesh.x0 + np.arange(mesh.nx + 1) * mesh.dx          # (nx+1,)
    yc = mesh.yc                                                # (ny,)
    XF, YF = np.meshgrid(xf, yc, indexing='xy')                # (ny, nx+1)
    u_face = -np.sin(np.pi * XF)**2 * np.sin(2.0 * np.pi * YF) * cos_t

    # y-faces: located at x = xc[i], y = y0 + j*dy  (cell-centre x)
    xc = mesh.xc                                                # (nx,)
    yf = mesh.y0 + np.arange(mesh.ny + 1) * mesh.dy            # (ny+1,)
    XF2, YF2 = np.meshgrid(xc, yf, indexing='xy')              # (ny+1, nx)
    v_face = np.sin(2.0 * np.pi * XF2) * np.sin(np.pi * YF2)**2 * cos_t

    return u_face, v_face
