"""
config.py
---------
ME 5351 HW2 - Central configuration for all tasks and test cases.

All physical parameters, numerical parameters, and I/O settings are
collected here as plain Python dicts / dataclasses.  Solvers import
this module to avoid hard-coding values.

Structure
~~~~~~~~~
DEFAULT_MESH_CFG    : Mesh resolution and domain size.
DEFAULT_DROP_CFG    : Drop geometry (centre, radius, interface thickness).
DEFAULT_SOLVER_CFG  : Time step, end time, CFL target, output frequency.
TASK_CONFIGS        : Per-task overrides (scheme labels, output directories).
TEST_CASE_CONFIGS   : Per-test-case parameter bundles.

Usage
-----
    from config import TASK_CONFIGS, TEST_CASE_CONFIGS
    cfg = {**DEFAULT_MESH_CFG, **DEFAULT_SOLVER_CFG, **TASK_CONFIGS[1]}
    run_drop_advection(task_id=1, cfg=cfg)

Notes
-----
- eps (interface half-thickness) is set to 1.5 * dx by default.  You may
  need to adjust this for coarser or finer meshes.
- CFL_target is used by the adaptive time-step controller (if implemented).
  For fixed-dt runs, dt is set explicitly.
- All paths use forward slashes for cross-platform compatibility.
- Mesh: paper used 50*50, and 512*512 for convergence test.
"""

import os

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results")

# ---------------------------------------------------------------------------
# Mesh defaults
# ---------------------------------------------------------------------------

DEFAULT_MESH_CFG: dict = {
    "nx": 256,       # cells in x-direction
    "ny": 256,       # cells in y-direction
    "Lx": 1.0,       # domain length in x
    "Ly": 1.0,       # domain length in y
    "x0": 0.0,       # domain origin x
    "y0": 0.0,       # domain origin y
}

# ---------------------------------------------------------------------------
# Drop geometry defaults
# ---------------------------------------------------------------------------

DEFAULT_DROP_CFG: dict = {
    # Test Case 1: drop at domain centre
    "drop_advection": {
        "x_drop": 0.5,
        "y_drop": 0.5,
        "R": 0.15,
    },
    # Test Case 2: drop at (0.5, 0.75) per Jain 2022
    "shear_flow": {
        "x_drop": 0.5,
        "y_drop": 0.75,
        "R": 0.15,
    },
}

# eps is set dynamically from mesh spacing: eps = eps_factor * dx
EPS_FACTOR: float = 1  # interface half-thickness in units of dx

# ---------------------------------------------------------------------------
# Solver defaults
# ---------------------------------------------------------------------------

DEFAULT_SOLVER_CFG: dict = {
    "CFL_target": 0.4,    # target CFL for adaptive dt (not used if dt given)
    "save_freq": 60,      # save every N time steps
}

# ---------------------------------------------------------------------------
# Test Case 1: Drop advection
# ---------------------------------------------------------------------------

DROP_ADVECTION_CFG: dict = {
    "test_case": "drop_advection",
    "velocity": "uniform",
    "U0": 5.0,            # advection velocity (problem statement)
    "V0": 0.0,
    "t_end": 1,         # drop travels U0 * t_end = 0.5 domain lengths
    "dt": 0.001,           # explicit Euler / RK4 time step
}

# ---------------------------------------------------------------------------
# Test Case 2: Oscillating shear flow
# ---------------------------------------------------------------------------

SHEAR_FLOW_CFG: dict = {
    "test_case": "shear_flow",
    "velocity": "shear",
    "T_period": 4.0,      # oscillation period (paper used 4)
    "t_end": 4.0,         # run for one full period
    "dt": 2.5e-4 ,        # aiming cfl ~ 0.25
}

# ---------------------------------------------------------------------------
# Per-task configuration
# ---------------------------------------------------------------------------

TASK_CONFIGS: dict[int, dict] = {
    1: {
        "task_id": 1,
        "label": "Task 1 - 1st-order Upwind + Euler",
        "spatial_scheme": "upwind",
        "temporal_scheme": "euler",
        "use_regularization": False,
        "output_dir": os.path.join(RESULTS_ROOT, "task1"),
    },
    2: {
        "task_id": 2,
        "label": "Task 2 - CDI + Upwind + Euler",
        "spatial_scheme": "upwind",
        "temporal_scheme": "euler",
        "use_regularization": True,
        "reg_method": "cdi",
        "output_dir": os.path.join(RESULTS_ROOT, "task2"),
    },
    3: {
        "task_id": 3,
        "label": "Task 3 - CDI + Central + RK4",
        "spatial_scheme": "central",
        "temporal_scheme": "rk4",
        "use_regularization": True,
        "reg_method": "cdi",
        "output_dir": os.path.join(RESULTS_ROOT, "task3"),
    },
    4: {
        "task_id": 4,
        "label": "Task 4 - ACDI + Skew-Symmetric + RK4",
        "spatial_scheme": "skew_symmetric",
        "temporal_scheme": "rk4",
        "use_regularization": True,
        "reg_method": "acdi",
        "output_dir": os.path.join(RESULTS_ROOT, "task4"),
    },
}
