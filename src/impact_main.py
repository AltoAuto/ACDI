"""
impact_main.py
--------------
CLI entry point for the 2D droplet impact simulation.

Physics
-------
A dense liquid drop falls onto a liquid pool.  The simulation uses
full incompressible Navier-Stokes with:
  - ACDI phase-field interface tracking (Jain 2022)
  - CSF surface tension (We-driven crown splash)
  - Chorin projection method for the velocity-pressure coupling

Non-dimensional parameters (D=1, U=1 by default):
  Re = 500  (mu1 = rho1 * U * D / Re = 2e-3)
  We = 100  (sigma = rho1 * U^2 * D / We = 0.01)

Usage
-----
    # Quick 64x64 test (no animation)
    python impact_main.py --nx 64 --t-end 1.0 --no-plot

    # Full run with animation
    python impact_main.py --nx 128 --save impact.gif

    # High-impact run (faster drop, more dramatic crown)
    python impact_main.py --nx 128 --drop-U 2.0 --re 1000 --save impact_fast.gif

Options
-------
    --nx INT          cells in x (default 128)
    --ny INT          cells in y (default: ny=nx, square grid for 4x4 domain)
    --t-end FLOAT     end time (default 3.0)
    --dt FLOAT        time step (default 1e-3)
    --re FLOAT        Reynolds number — overrides mu1 (default: use config mu1)
    --we FLOAT        Weber number   — overrides sigma (default: use config sigma)
    --drop-R FLOAT    drop radius (default 0.5)
    --drop-cx FLOAT   drop x-centre (default 2.0)
    --drop-cy FLOAT   drop y-centre (default 3.0)
    --drop-U FLOAT    initial drop downward velocity (default 1.0)
    --pool-y FLOAT    pool surface y-position (default 1.5)
    --eps-factor FLOAT  eps = eps_factor * dx (default 1.5)
    --save PATH       output GIF path (default impact.gif)
    --fps INT         animation fps (default 15)
    --save-freq INT   snapshot every N steps (default 10)
    --no-plot         skip animation, print stats only
"""

import sys
import os
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from impact.config import IMPACT_DEFAULTS
from impact.solver import run_impact
from impact.plotting import animate_impact


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="2D Droplet Impact Simulation (NS + ACDI phase-field)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--nx",         type=int,   default=None)
    p.add_argument("--ny",         type=int,   default=None)
    p.add_argument("--t-end",      type=float, default=None, dest="t_end")
    p.add_argument("--dt",         type=float, default=None)
    p.add_argument("--re",         type=float, default=None,
                   help="Reynolds number (overrides mu1)")
    p.add_argument("--we",         type=float, default=None,
                   help="Weber number (overrides sigma)")
    p.add_argument("--drop-R",     type=float, default=None, dest="drop_R")
    p.add_argument("--drop-cx",    type=float, default=None, dest="drop_cx")
    p.add_argument("--drop-cy",    type=float, default=None, dest="drop_cy")
    p.add_argument("--drop-U",     type=float, default=None, dest="drop_U")
    p.add_argument("--pool-y",     type=float, default=None, dest="pool_y")
    p.add_argument("--eps-factor", type=float, default=None, dest="eps_factor")
    p.add_argument("--save",       type=str,   default=None)
    p.add_argument("--fps",        type=int,   default=None)
    p.add_argument("--save-freq",  type=int,   default=None, dest="save_freq")
    p.add_argument("--no-plot",    action="store_true")
    return p.parse_args(argv)


def _pick(val, default):
    """Return val if not None, else default."""
    return val if val is not None else default


def main(argv=None):
    args = parse_args(argv)
    d    = IMPACT_DEFAULTS

    # Build cfg, CLI args override defaults
    cfg = dict(d)   # start from defaults
    cfg["nx"]         = _pick(args.nx,         d["nx"])
    cfg["ny"]         = _pick(args.ny,         cfg["nx"])   # square by default
    cfg["t_end"]      = _pick(args.t_end,      d["t_end"])
    cfg["dt"]         = _pick(args.dt,         d["dt"])
    cfg["drop_R"]     = _pick(args.drop_R,     d["drop_R"])
    cfg["drop_cx"]    = _pick(args.drop_cx,    d["drop_cx"])
    cfg["drop_cy"]    = _pick(args.drop_cy,    d["drop_cy"])
    cfg["drop_U"]     = _pick(args.drop_U,     d["drop_U"])
    cfg["pool_y"]     = _pick(args.pool_y,     d["pool_y"])
    cfg["eps_factor"] = _pick(args.eps_factor, d["eps_factor"])
    cfg["fps"]        = _pick(args.fps,        d["fps"])
    cfg["save_freq"]  = _pick(args.save_freq,  d["save_freq"])
    cfg["output"]     = _pick(args.save,       d["output"])

    # Re / We overrides
    if args.re is not None:
        cfg["mu1"] = cfg["rho1"] * cfg["drop_U"] / args.re
        cfg["mu2"] = cfg["mu1"] / 50.0
    if args.we is not None:
        cfg["sigma"] = cfg["rho1"] * cfg["drop_U"] ** 2 / args.we

    # Print run summary
    nx, ny = cfg["nx"], cfg["ny"]
    dx     = cfg["Lx"] / nx
    Re_eff = cfg["rho1"] * cfg["drop_U"] / cfg["mu1"]
    We_eff = cfg["rho1"] * cfg["drop_U"] ** 2 / cfg["sigma"]

    print("=" * 64)
    print("Droplet Impact  |  NS + ACDI phase-field")
    print("=" * 64)
    print(f"  Grid       : {nx} x {ny}  "
          f"(Lx={cfg['Lx']}, Ly={cfg['Ly']}, dx={dx:.4e})")
    print(f"  Re         : {Re_eff:.1f}  "
          f"(mu1={cfg['mu1']:.2e}, mu2={cfg['mu2']:.2e})")
    print(f"  We         : {We_eff:.1f}  (sigma={cfg['sigma']:.2e})")
    print(f"  g          : {cfg['g']}  (gravity, downward)")
    print(f"  eps_factor : {cfg['eps_factor']}  (eps={cfg['eps_factor']*dx:.4e})")
    print(f"  drop       : R={cfg['drop_R']}  "
          f"cx={cfg['drop_cx']}  cy={cfg['drop_cy']}  "
          f"U={cfg['drop_U']}")
    print(f"  pool_y     : {cfg['pool_y']}")
    print(f"  t_end      : {cfg['t_end']}  |  dt : {cfg['dt']}")
    print(f"  save_freq  : {cfg['save_freq']}  |  output : {cfg['output']}")
    print("=" * 64 + "\n")

    # Run simulation
    history = run_impact(cfg)

    print(f"\nSimulation complete. {len(history)} snapshots collected.")

    # Animation
    if not args.no_plot:
        from core.mesh import Mesh
        mesh = Mesh(nx=nx, ny=cfg["ny"], Lx=cfg["Lx"], Ly=cfg["Ly"])
        animate_impact(
            history, mesh,
            save_path=cfg["output"],
            fps=cfg["fps"],
            drop_U=cfg["drop_U"],
        )
    else:
        print("--no-plot: skipping animation.")


if __name__ == "__main__":
    main()
