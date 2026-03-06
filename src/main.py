"""
main.py
-------
ME 5351 HW2 - Entry point for running all tasks and test cases.

Usage
-----
    # Run all four tasks on both test cases
    python main.py

    # Run a specific task on a specific test case
    python main.py --task 4 --case shear

    # Run a mesh refinement convergence study
    python main.py --convergence --task 3

Command-line arguments
----------------------
  --task  {1,2,3,4}          Task to run (default: all)
  --case  {drop,shear}       Test case to run (default: both)
  --convergence              Run convergence study instead of single run
  --nx    INT                Override mesh resolution (default from config)
  --dt    FLOAT              Override time step (default from config)
  --eps   FLOAT              Override interface thickness (default: 1.5*dx)
  --animate                  Save animation to results/ directory
  --no-plot                  Suppress figure display (useful for batch runs)

Output
------
  - Phi snapshots saved as .npy files in results/taskN/
  - PNG figures saved in results/taskN/
  - Error tables printed to stdout and saved as results/errors.txt

Workflow
--------
1. Parse CLI arguments and build configuration dict.
2. Construct Mesh and initial conditions.
3. Dispatch to the appropriate solver(s) via test_cases module.
4. Post-process: compute errors, generate plots, print table.
"""

import argparse
import os
import numpy as np

from config import (
    DEFAULT_MESH_CFG,
    DEFAULT_SOLVER_CFG,
    DEFAULT_DROP_CFG,
    DROP_ADVECTION_CFG,
    SHEAR_FLOW_CFG,
    TASK_CONFIGS,
    EPS_FACTOR,
    RESULTS_ROOT,
)
from test_cases import run_drop_advection, run_shear_flow
from postprocessing import print_error_table, plot_phi_field, plot_interface_contour


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ME 5351 HW2 - Phase-field advection solver (CDI/ACDI)"
    )
    parser.add_argument(
        "--task", type=int, choices=[1, 2, 3, 4], default=None,
        help="Task to run (1-4).  Defaults to all tasks."
    )
    parser.add_argument(
        "--case", choices=["drop", "shear"], default=None,
        help="Test case to run.  Defaults to both."
    )
    parser.add_argument(
        "--convergence", action="store_true",
        help="Run mesh refinement convergence study."
    )
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--eps", type=float, default=None)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def build_cfg(
    task_id: int,
    test_case_cfg: dict,
    args: argparse.Namespace,
) -> dict:
    """Merge default, test-case, and task configs; apply CLI overrides.

    Parameters
    ----------
    task_id : int
        Task number (1-4).
    test_case_cfg : dict
        Test-case specific parameters (DROP_ADVECTION_CFG or SHEAR_FLOW_CFG).
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    cfg : dict
        Merged configuration dictionary ready to pass to a solver.
    """
    cfg = {}
    cfg.update(DEFAULT_MESH_CFG)
    cfg.update(DEFAULT_SOLVER_CFG)
    cfg.update(test_case_cfg)
    cfg.update(TASK_CONFIGS[task_id])

    # Add drop geometry based on test case
    tc = test_case_cfg.get("test_case", "drop_advection")
    if tc == "drop_advection":
        cfg.update(DEFAULT_DROP_CFG["drop_advection"])
    elif tc == "shear_flow":
        cfg.update(DEFAULT_DROP_CFG["shear_flow"])

    # CLI overrides
    if args.nx is not None:
        cfg["nx"] = args.nx
        cfg["ny"] = args.nx  # square mesh

    if args.dt is not None:
        cfg["dt"] = args.dt

    # Compute eps from dx
    dx = cfg["Lx"] / cfg["nx"]
    if args.eps is not None:
        cfg["eps"] = args.eps
    else:
        cfg["eps"] = EPS_FACTOR * dx

    # Compute R from config
    if "R" not in cfg:
        cfg["R"] = 0.15

    # Ensure output directory exists
    os.makedirs(cfg["output_dir"], exist_ok=True)

    return cfg


def run_convergence_study(task_id: int, test_case: str, args: argparse.Namespace) -> None:
    """Run the solver at multiple mesh resolutions and plot convergence.

    Mesh sizes: nx in [32, 64, 128, 256] (square meshes).
    dt scaled as dt ~ dx to keep CFL constant across refinements.

    Parameters
    ----------
    task_id : int
    test_case : str  -- 'drop' or 'shear'
    args : argparse.Namespace
    """

    from postprocessing import plot_convergence

    test_case_cfg = DROP_ADVECTION_CFG if test_case == "drop" else SHEAR_FLOW_CFG
    run_fn = run_drop_advection if test_case == "drop" else run_shear_flow

    nx_list = [32, 64, 128, 256]
    dx_list = []
    errors_l1 = []
    errors_l2 = []
    errors_linf = []

    # Reference dt for scaling
    base_dt = test_case_cfg["dt"]
    base_nx = DEFAULT_MESH_CFG["nx"]

    for nx in nx_list:
        print(f"  Convergence: nx={nx} ...", end=" ", flush=True)
        # Scale dt to maintain constant CFL
        args_copy = argparse.Namespace(**vars(args))
        args_copy.nx = nx
        scale = base_nx / nx
        args_copy.dt = base_dt * scale
        args_copy.eps = None  # auto-compute from dx

        cfg = build_cfg(task_id, test_case_cfg, args_copy)
        _, _, metrics = run_fn(task_id, cfg)

        dx_list.append(cfg["Lx"] / nx)
        errors_l1.append(metrics["L1"])
        errors_l2.append(metrics["L2"])
        errors_linf.append(metrics["Linf"])
        print(f"L2={metrics['L2']:.4e}")

    label = TASK_CONFIGS[task_id]["label"]
    error_dict = {
        f"L1 ({label})": errors_l1,
        f"L2 ({label})": errors_l2,
        f"Linf ({label})": errors_linf,
    }

    save_path = os.path.join(RESULTS_ROOT, f"convergence_task{task_id}_{test_case}.png")
    plot_convergence(dx_list, error_dict, title=f"Convergence: {label} ({test_case})",
                     save_path=save_path)
    print(f"  Convergence plot saved to {save_path}")


def main() -> None:
    """Main driver: parse arguments, run tasks, post-process results."""
    args = parse_args()

    tasks = [args.task] if args.task else [1, 2, 3, 4]
    cases = [args.case] if args.case else ["drop", "shear"]

    if args.convergence:
        for task_id in tasks:
            for case in cases:
                print(f"\n=== Convergence Study: Task {task_id}, {case} ===")
                run_convergence_study(task_id, case, args)
        return

    all_errors: dict[str, dict[str, dict[str, float]]] = {}
    all_final_phi: dict[str, dict[str, np.ndarray]] = {}
    all_meshes: dict[str, object] = {}

    for case in cases:
        test_case_cfg = DROP_ADVECTION_CFG if case == "drop" else SHEAR_FLOW_CFG
        case_errors: dict[str, dict[str, float]] = {}
        case_phi: dict[str, np.ndarray] = {}

        for task_id in tasks:
            cfg = build_cfg(task_id, test_case_cfg, args)
            label = TASK_CONFIGS[task_id]["label"]
            print(f"\n--- Running {label} on {case} test case ---")

            run_fn = run_drop_advection if case == "drop" else run_shear_flow
            phi_history, t_history, metrics = run_fn(task_id, cfg)

            case_errors[label] = metrics
            case_phi[label] = phi_history[-1]
            all_meshes[case] = cfg["mesh"]

            if not args.no_plot:
                plot_phi_field(
                    phi=phi_history[-1],
                    mesh=cfg["mesh"],
                    t=t_history[-1],
                    title=f"{label} | {case} | t = {t_history[-1]:.3f}",
                    save_path=os.path.join(cfg["output_dir"], f"{case}_final.png"),
                )

            if args.animate:
                from postprocessing import animate_phi
                anim_path = os.path.join(cfg["output_dir"], f"{case}_anim.gif")
                animate_phi(phi_history, t_history, cfg["mesh"], anim_path, fps=10)
                print(f"  Animation saved to {anim_path}")

        all_errors[case] = case_errors
        all_final_phi[case] = case_phi
        print(f"\n{'='*60}")
        print_error_table(case_errors, title=f"{case.upper()} Test Case Errors")

    # Compare interface contours across tasks
    for case in cases:
        if not args.no_plot and len(tasks) > 1 and case in all_final_phi:
            save_path = os.path.join(RESULTS_ROOT, f"{case}_contour_comparison.png")
            mesh = all_meshes[case]
            t_final = DROP_ADVECTION_CFG["t_end"] if case == "drop" else SHEAR_FLOW_CFG["t_end"]
            plot_interface_contour(
                all_final_phi[case], mesh, t=t_final,
                title=f"Interface comparison ({case})",
                save_path=save_path,
            )
            print(f"Contour comparison saved to {save_path}")


if __name__ == "__main__":
    main()
