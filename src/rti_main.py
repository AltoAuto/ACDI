"""
rti_main.py
-----------
CLI entry point for the Rayleigh-Taylor instability (RTI) extension.

What RTI actually is
--------------------
RTI is the instability of a HEAVY FLUID on top of a LIGHT FLUID under
gravity.  The dense fluid sinks as mushroom-shaped "fingers" while the
lighter fluid rises as round "bubbles".  This IS the physics of a dense
droplet sinking through lighter fluid — the mushroom cap at the finger
tip IS the "drop" shape.  It is NOT the same as droplet impact on a
liquid surface (which requires surface tension + inertia, not modelled here).

Initial conditions
------------------
  drop (default)
      An isolated circular blob of heavy fluid in a tall light-fluid domain.
      The bottom surface of the blob is RTI-unstable: fingers grow DOWNWARD
      from the base, creating the classic tentacle/mushroom pattern of a
      dense drop sinking.  Use Lx=1, Ly=2 (default for this IC) so the
      fingers have space to travel before the periodic boundary wraps them.
      Defaults: g_eff=100, R=0.12*Ly, cy=0.72*Ly, Ly=2.0, t_end=3.0

  mound
      A flat heavy/light interface with a large Gaussian tongue of heavy
      fluid pointing downward.  Shows a SINGLE dominant mushroom growing from
      an initially flat interface.
      Defaults: g_eff=100, A0=0.18*Ly, sigma=0.12*Lx, t_end=2.5

  flat
      Classic flat interface + small cosine perturbation(s).  Good for
      growth-rate studies.
      Defaults: g_eff=50, A0=0.01*Ly, n_modes=1, t_end=1.5

Animation panels (4-panel layout)
----------------------------------
  Top row    : phi field — CDI (left) | ACDI (right)
  Bottom row : |grad phi| (interface sharpness) — CDI | ACDI
               ACDI stays bright/thin; CDI dims and widens over time.
  Subtitle   : diffuse-cell count (0.05<phi<0.95) per method.

Usage
-----
    # Drop IC — most visually dramatic (tall domain, isolated heavy blob)
    python rti_main.py --save rti_drop.gif

    # Mound IC
    python rti_main.py --ic mound --save rti_mound.gif

    # Multi-mode flat IC
    python rti_main.py --ic flat --n-modes 3 --t-end 2.0 --save rti_flat.gif

    # Quick test
    python rti_main.py --nx 64 --t-end 1.0 --no-plot

Options
-------
    --ic {drop,mound,flat}   initial condition (default: drop)
    --nx INT                 grid cells in x (default 128)
    --ny INT                 grid cells in y (default: IC-specific)
    --Lx FLOAT               domain width  (default 1.0)
    --Ly FLOAT               domain height (IC default: drop=2.0, else=1.0)
    --t-end FLOAT            end time      (IC default: drop=3.0, mound=2.5, flat=1.5)
    --dt FLOAT               timestep      (default 5e-4)
    --g-eff FLOAT            2*g*At/nu     (IC default: drop/mound=100, flat=50)
    --a0 FLOAT               IC amplitude as fraction of Ly (IC-specific default)
    --n-modes INT            cosine modes for flat IC
    --mound-sigma FLOAT      Gaussian width as fraction of Lx (mound IC)
    --drop-R FLOAT           droplet radius as fraction of min(Lx,Ly) (default 0.12)
    --drop-cy FLOAT          droplet y-centre as fraction of Ly (default 0.72)
    --eps-factor FLOAT       eps = eps_factor * dx (default 1.5)
    --save PATH              output file (IC default: rti_drop/mound/flat.gif)
    --fps INT                animation fps (default 15)
    --save-freq INT          snapshot interval in steps (default 20)
    --no-plot                skip animation, only print stats
"""

import sys
import os
import argparse

_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from core.mesh import Mesh
from rti.initial import rti_initial_condition, mound_initial_condition, drop_initial_condition
from rti.solvers import run_rti_cdi, run_rti_acdi
from rti.plotting import animate_comparison

# Per-IC sensible defaults — all overridable via CLI
_IC_DEFAULTS = {
    "drop": {
        "Lx":           1.0,
        "Ly":           2.0,   # tall domain: fingers fall further before wrapping
        "ny_factor":    2,     # ny = ny_factor * nx (square pixels for tall domain)
        "g_eff":        100.0,
        "t_end":        3.0,
        "A0_factor":    None,  # not used for drop IC
        "n_modes":      1,
        "mound_sigma":  0.12,
        "drop_R":       0.12,  # radius = 0.12 * min(Lx, Ly)
        "drop_cy":      0.72,  # centre-y = 0.72 * Ly
        "output":       "rti_drop.gif",
    },
    "mound": {
        "Lx":           1.0,
        "Ly":           1.0,
        "ny_factor":    1,
        "g_eff":        100.0,
        "t_end":        2.5,
        "A0_factor":    0.18,
        "n_modes":      1,
        "mound_sigma":  0.12,
        "drop_R":       0.12,
        "drop_cy":      0.72,
        "output":       "rti_mound.gif",
    },
    "flat": {
        "Lx":           1.0,
        "Ly":           1.0,
        "ny_factor":    1,
        "g_eff":        50.0,
        "t_end":        1.5,
        "A0_factor":    0.01,
        "n_modes":      1,
        "mound_sigma":  0.12,
        "drop_R":       0.12,
        "drop_cy":      0.72,
        "output":       "rti_flat.gif",
    },
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Rayleigh-Taylor Instability — CDI vs ACDI (4-panel animation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "IC defaults:\n"
            "  drop : Ly=2.0, ny=2*nx, g_eff=100, R=0.12*Ly, cy=0.72*Ly, t_end=3.0\n"
            "  mound: Ly=1.0, g_eff=100, A0=0.18*Ly, sigma=0.12*Lx, t_end=2.5\n"
            "  flat : Ly=1.0, g_eff=50,  A0=0.01*Ly, n_modes=1,     t_end=1.5\n"
        ),
    )
    p.add_argument("--ic",          choices=["drop", "mound", "flat"], default="drop")
    p.add_argument("--nx",          type=int,   default=128)
    p.add_argument("--ny",          type=int,   default=None)
    p.add_argument("--Lx",          type=float, default=None)
    p.add_argument("--Ly",          type=float, default=None)
    p.add_argument("--t-end",       type=float, default=None, dest="t_end")
    p.add_argument("--dt",          type=float, default=5e-4)
    p.add_argument("--g-eff",       type=float, default=None, dest="g_eff")
    p.add_argument("--a0",          type=float, default=None)
    p.add_argument("--n-modes",     type=int,   default=None, dest="n_modes")
    p.add_argument("--mound-sigma", type=float, default=None, dest="mound_sigma")
    p.add_argument("--drop-R",      type=float, default=None, dest="drop_R",
                   help="droplet radius as fraction of min(Lx,Ly)")
    p.add_argument("--drop-cy",     type=float, default=None, dest="drop_cy",
                   help="droplet y-centre as fraction of Ly")
    p.add_argument("--eps-factor",  type=float, default=1.5, dest="eps_factor")
    p.add_argument("--save",        type=str,   default=None)
    p.add_argument("--fps",         type=int,   default=15)
    p.add_argument("--save-freq",   type=int,   default=20, dest="save_freq")
    p.add_argument("--no-plot",     action="store_true")
    return p.parse_args(argv)


def _pick(arg, default):
    """Return arg if set by user (not None), else default."""
    return arg if arg is not None else default


def main(argv=None):
    args = parse_args(argv)
    d = _IC_DEFAULTS[args.ic]

    # Resolve all parameters, falling back to IC-specific defaults
    Lx          = _pick(args.Lx,          d["Lx"])
    Ly          = _pick(args.Ly,          d["Ly"])
    nx          = args.nx
    ny          = _pick(args.ny,          d["ny_factor"] * nx)
    g_eff       = _pick(args.g_eff,       d["g_eff"])
    t_end       = _pick(args.t_end,       d["t_end"])
    n_modes     = _pick(args.n_modes,     d["n_modes"])
    mound_sigma = _pick(args.mound_sigma, d["mound_sigma"])
    drop_R_f    = _pick(args.drop_R,      d["drop_R"])
    drop_cy_f   = _pick(args.drop_cy,     d["drop_cy"])
    save_path   = _pick(args.save,        d["output"])

    mesh = Mesh(nx=nx, ny=ny, Lx=Lx, Ly=Ly)
    eps  = args.eps_factor * mesh.dx

    print("=" * 64)
    print(f"Rayleigh-Taylor Instability  ({args.ic} IC)  |  CDI vs ACDI")
    print("=" * 64)
    print(f"  Grid      : {nx} x {ny}  "
          f"(Lx={Lx}, Ly={Ly}, dx={mesh.dx:.4e}, dy={mesh.dy:.4e})")
    print(f"  eps       : {eps:.4e}  ({args.eps_factor}*dx)")
    print(f"  g_eff     : {g_eff}  [1/(m*s)]")
    print(f"  t_end     : {t_end}  |  dt : {args.dt}")

    # Build initial condition
    if args.ic == "drop":
        R  = drop_R_f  * min(Lx, Ly)
        cx = Lx / 2.0
        cy = drop_cy_f * Ly
        print(f"  R         : {R:.4f}  ({drop_R_f}*min(Lx,Ly))")
        print(f"  centre    : ({cx:.3f}, {cy:.3f})")
        phi0 = drop_initial_condition(mesh, R=R, cx=cx, cy=cy, eps=eps)

    elif args.ic == "mound":
        A0_factor = _pick(args.a0, d["A0_factor"])
        A0        = A0_factor * Ly
        sigma     = mound_sigma * Lx
        print(f"  A0        : {A0:.4e}  ({A0_factor}*Ly)")
        print(f"  sigma     : {sigma:.4e}  ({mound_sigma}*Lx)")
        phi0 = mound_initial_condition(mesh, A0=A0, sigma=sigma, eps=eps)

    else:  # flat
        A0_factor = _pick(args.a0, d["A0_factor"])
        A0        = A0_factor * Ly
        print(f"  A0        : {A0:.4e}  ({A0_factor}*Ly)")
        print(f"  n_modes   : {n_modes}")
        phi0 = rti_initial_condition(mesh, A0=A0, n_modes=n_modes, eps=eps)

    print("=" * 64)
    mass0 = phi0.sum() * mesh.dx * mesh.dy
    print(f"Initial phi: [{phi0.min():.4f}, {phi0.max():.4f}]  "
          f"mass={mass0:.6f}\n")

    cfg = {
        "mesh":      mesh,
        "phi0":      phi0,
        "eps":       eps,
        "g_eff":     g_eff,
        "t_end":     t_end,
        "dt":        args.dt,
        "save_freq": args.save_freq,
    }

    cdi_hist,  t_hist_cdi  = run_rti_cdi(cfg)
    acdi_hist, t_hist_acdi = run_rti_acdi(cfg)

    n = min(len(cdi_hist), len(acdi_hist))
    t_hist = t_hist_cdi[:n]
    print(f"Snapshots  : CDI={len(cdi_hist)}, ACDI={len(acdi_hist)}, using {n}")

    if not args.no_plot:
        animate_comparison(
            cdi_hist[:n], acdi_hist[:n], t_hist,
            mesh=mesh,
            save_path=save_path,
            fps=args.fps,
        )
    else:
        print("--no-plot: skipping animation.")


if __name__ == "__main__":
    main()
