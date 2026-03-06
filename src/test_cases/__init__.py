"""
test_cases/__init__.py
----------------------
ME 5351 HW2 - Test case factory.

Provides high-level entry points for the two standard validation test cases:

1. drop_advection : Circular drop in a uniform velocity field (u = 5, v = 0).
   The drop should translate at constant speed with minimal deformation.
   Error at final time is compared to the initial condition (exact solution).

2. shear_flow : Circular drop in an oscillating shear flow (LeVeque 1996).
   The drop deforms and then returns to its initial shape at t = T.
   The error at t = T relative to the IC quantifies reversibility.

Each test case module exports a `run(task_id, cfg)` function that dispatches
to the appropriate solver and returns (phi_history, t_history, metrics).
"""

from .drop_advection import run_drop_advection
from .shear_flow import run_shear_flow
