"""
solvers/__init__.py
-------------------
ME 5351 HW2 - Solver registry.

Each sub-module implements one of the four homework tasks as a self-contained
solver that accepts a configuration dictionary and returns a time-history of
the volume-fraction field.

Task 1 : task1_upwind   -- 1st-order upwind + explicit Euler (plain advection)
Task 2 : task2_cdi_1st  -- CDI + 1st-order upwind + explicit Euler
Task 3 : task3_cdi_2nd_rk4 -- CDI + 2nd-order central + RK4
Task 4 : task4_acdi     -- ACDI (Jain 2022 Eq. 20-21) + 2nd-order + RK4
"""

from .task1_upwind import run_task1
from .task2_cdi_1st import run_task2
from .task3_cdi_2nd_rk4 import run_task3
from .task4_acdi import run_task4
