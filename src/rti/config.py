"""
rti/config.py
-------------
Default configuration parameters for the Rayleigh-Taylor instability simulation.
"""

RTI_DEFAULTS = {
    "nx": 512,
    "ny": 512,
    "Lx": 1.0,
    "Ly": 1.0,
    "eps_factor": 1.5,       # eps = eps_factor * dx
    "A0_factor": 0.01,       # perturbation amplitude = A0_factor * Lx
    "n_modes": 1,            # number of cosine modes in perturbation
    "g_eff": 50.0,           # lumped parameter 2*g*At/nu [1/(m*s)]
    "t_end": 1.5,
    "dt": 5e-4,
    "save_freq": 20,
    "fps": 15,
    "output": "rti_comparison.gif",
}
