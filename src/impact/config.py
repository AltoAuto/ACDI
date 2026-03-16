"""
impact/config.py
----------------
Default configuration for the 2D droplet impact simulation.

Non-dimensional parameters (D=1, U=1):
  Re = rho1 * U * D / mu1 = 500  ->  mu1 = 1/Re = 2e-3
  We = rho1 * U^2 * D / sigma = 100  ->  sigma = 1/We = 0.01
"""

IMPACT_DEFAULTS = {
    "nx": 128, "ny": 128,
    "Lx": 4.0, "Ly": 4.0,
    "rho1": 1.0,   "rho2": 0.01,      # liquid / gas density
    "mu1":  2e-3,  "mu2":  4e-5,      # liquid / gas viscosity
    "sigma": 0.01,                     # surface tension (We=100)
    "g":     0.5,                      # gravity (non-dimensional, downward)
    "eps_factor": 1.0,                 # eps = eps_factor * dx  (ACDI: sharper = less kappa noise)
    "drop_R":  0.5,                    # drop radius (D/2)
    "drop_cx": 2.0,                    # drop x-centre
    "drop_cy": 3.0,                    # drop y-centre
    "drop_U":  1.0,                    # initial downward velocity
    "pool_y":  1.5,                    # pool surface y-position
    "t_end":   3.0,
    "dt":      1e-3,
    "save_freq": 10,
    "fps": 15,
    "output": "impact.gif",
}
