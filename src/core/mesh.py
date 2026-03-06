"""
core/mesh.py
------------
ME 5351 HW2 - Structured 2-D Cartesian mesh for finite-volume phase-field solver.

Provides the `Mesh` class that holds:
  - Cell-centre coordinates (xc, yc)
  - Face coordinates (xf, yf) for east/west/north/south faces
  - Grid spacings (dx, dy) and domain extents (Lx, Ly)
  - Cell count (nx, ny) and total cell number N = nx * ny
  - Helper index maps for vectorised stencil access (periodic BC by default)

Conventions
-----------
  - (i, j) indexing:  i -> x-direction (columns), j -> y-direction (rows)
  - Flattened index:  k = j * nx + i
  - Ghost cells are handled via numpy roll (periodic) or explicit padding for
    other boundary types.

Reference
---------
  Jain, S. S. (2022). Accurate conservative phase-field method for simulation
  of two-phase flows. J. Comput. Phys., 469, 111529.
"""

import numpy as np


class Mesh:
    """Uniform 2-D Cartesian mesh for the phase-field advection solver.

    Parameters
    ----------
    nx : int
        Number of cells in the x-direction.
    ny : int
        Number of cells in the y-direction.
    Lx : float
        Domain length in the x-direction.
    Ly : float
        Domain length in the y-direction.
    x0 : float, optional
        x-coordinate of the domain origin (default 0.0).
    y0 : float, optional
        y-coordinate of the domain origin (default 0.0).
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        Lx: float,
        Ly: float,
        x0: float = 0.0,
        y0: float = 0.0,
    ) -> None:
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.x0 = x0
        self.y0 = y0
        self.dx = Lx / nx
        self.dy = Ly / ny
        self.N = nx * ny

        # Cell-centre coordinates (1-D)
        self._xc = x0 + (np.arange(nx) + 0.5) * self.dx
        self._yc = y0 + (np.arange(ny) + 0.5) * self.dy

        # 2-D meshgrids (indexing='xy' gives shape (ny, nx))
        self._XC, self._YC = np.meshgrid(self._xc, self._yc, indexing='xy')

    # ------------------------------------------------------------------
    # Properties (computed once, stored as numpy arrays)
    # ------------------------------------------------------------------

    @property
    def xc(self) -> np.ndarray:
        """1-D array of cell-centre x-coordinates, shape (nx,)."""
        return self._xc

    @property
    def yc(self) -> np.ndarray:
        """1-D array of cell-centre y-coordinates, shape (ny,)."""
        return self._yc

    @property
    def XC(self) -> np.ndarray:
        """2-D meshgrid of cell-centre x-coords, shape (ny, nx)."""
        return self._XC

    @property
    def YC(self) -> np.ndarray:
        """2-D meshgrid of cell-centre y-coords, shape (ny, nx)."""
        return self._YC

    # ------------------------------------------------------------------
    # Neighbour index helpers (periodic boundary conditions)
    # ------------------------------------------------------------------

    def i_east(self, i: np.ndarray) -> np.ndarray:
        """Return east-neighbour column indices (periodic)."""
        return (i + 1) % self.nx

    def i_west(self, i: np.ndarray) -> np.ndarray:
        """Return west-neighbour column indices (periodic)."""
        return (i - 1) % self.nx

    def j_north(self, j: np.ndarray) -> np.ndarray:
        """Return north-neighbour row indices (periodic)."""
        return (j + 1) % self.ny

    def j_south(self, j: np.ndarray) -> np.ndarray:
        """Return south-neighbour row indices (periodic)."""
        return (j - 1) % self.ny

    def __repr__(self) -> str:
        return (f"Mesh(nx={self.nx}, ny={self.ny}, "
                f"Lx={self.Lx}, Ly={self.Ly}, "
                f"dx={self.dx:.4e}, dy={self.dy:.4e})")
