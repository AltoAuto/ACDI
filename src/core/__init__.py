"""
core/__init__.py
----------------
ME 5351 HW2 - Phase-field advection (CDI/ACDI) based on Jain (2022).

Exposes the core building blocks shared across all four tasks:
  - Mesh construction and indexing
  - Initial conditions (circular drop profile)
  - Prescribed velocity fields (uniform and oscillating shear)
  - Flux discretisation schemes (upwind, central)
  - Time integration routines (Euler, RK4)
  - Interface regularisation / re-initialisation kernels (CDI, ACDI)
"""

from .mesh import Mesh
from .initial_conditions import circular_drop
from .velocity_fields import uniform_velocity, shear_flow_velocity
from .flux_schemes import upwind_flux, central_flux
from .time_integration import euler_step, rk4_step
from .regularization import cdi_regularization, acdi_regularization
