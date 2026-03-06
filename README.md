# Phase-Field Interface Advection: CDI & ACDI Methods

A 2D finite-volume solver for accurate advection of volume fraction in two-phase flows, implementing and comparing four numerical schemes — from basic upwind to the Accurate Conservative Diffuse-Interface (ACDI) method proposed by Jain (2022).

---

## Overview

Accurately transporting a sharp interface between two fluid phases (e.g. a droplet in a gas) is a fundamental challenge in computational fluid dynamics. Naive advection schemes introduce numerical diffusion that smears the interface, while high-order schemes can be unstable without dissipation. Phase-field (diffuse-interface) methods address this by introducing controlled regularisation terms that balance diffusion and sharpening at the interface.

This project implements four progressively refined solvers on a uniform 2D Cartesian grid and benchmarks them against canonical test cases from the literature.

---

## Methods Implemented

| # | Scheme | Spatial | Temporal | Regularisation |
|---|--------|---------|----------|----------------|
| 1 | Plain scalar advection | 1st order upwind | Forward Euler | None |
| 2 | CDI — 1st order | 1st order upwind | Forward Euler | CDI (Chiu & Lin 2011) |
| 3 | CDI — 2nd order | Central difference | RK4 | CDI |
| 4 | **ACDI** | Central (skew-symmetric) | RK4 | ACDI (Jain 2022) |

**CDI** (Conservative Diffuse-Interface) adds a diffusion term and an interface-sharpening term to the transport equation, keeping φ bounded between 0 and 1:

$$\frac{\partial \phi}{\partial t} + \nabla \cdot (\mathbf{u}\phi) = \nabla \cdot \left\{ \Gamma \left\[ \epsilon \nabla\phi - \phi(1-\phi)\frac{\nabla\phi}{|\nabla\phi|} \right] \right\}$$

**ACDI** improves on CDI by replacing the nonlinear sharpening term with one expressed in terms of a signed-distance-like variable ψ, eliminating artificial grid-alignment distortion of the interface:

$$\frac{\partial \phi}{\partial t} + \nabla \cdot (\mathbf{u}\phi) = \nabla \cdot \left\{ \Gamma \left\[ \epsilon \nabla\phi - \frac{1}{4}\left(1 - \tanh^2\!\left(\frac{\psi}{2\epsilon}\right)\right)\frac{\nabla\psi}{|\nabla\psi|} \right] \right\}$$

---

## Test Cases

### 1. Drop Advection
A circular drop is advected across a periodic domain with a uniform velocity field **u** = 5**i** for 5 flow-through times. The exact solution is the initial shape translated — any deformation is purely numerical error.

### 2. Drop in Oscillating Shear Flow
A circular drop is deformed by a time-reversing shear velocity field:

$$u = -\sin^2(\pi x)\sin(2\pi y)\cos\!\left(\frac{\pi t}{T}\right), \quad v = \sin(2\pi x)\sin^2(\pi y)\cos\!\left\(\frac{\pi t}{T}\right)$$

The flow reverses at t = T/2, so the drop should recover its original shape at t = T. Error in the recovered shape quantifies numerical dissipation and interface distortion.

---

## Project Structure

```
src/
├── core/               # Mesh, ICs, velocity fields, flux schemes, time integration, regularisation
├── solvers/            # One solver module per method (tasks 1–4)
├── test_cases/         # Test case setup and error computation
├── postprocessing/     # Plotting and analysis utilities
├── results/            # Output figures and data (per task)
├── config.py           # Simulation parameters
└── main.py             # Entry point
```

---

## Key Parameters

| Parameter | Symbol | Role |
|-----------|--------|------|
| Interface thickness | ε | Controls interface width; smaller = sharper but stricter CFL |
| Velocity scale | Γ | Controls regularisation speed; must satisfy Γ ≥ \|**u**\|_max (ACDI) |
| Grid size | N | Uniform N×N Cartesian grid over [0,1]² |
| Time step | Δt | Constrained by CFL and viscous stability criterion |

---

## Reference

> Jain, S. S. (2022). *Accurate conservative phase-field method for simulation of two-phase flows.* arXiv:2203.05802

> Chiu, P.-H., & Lin, Y.-T. (2011). *A conservative phase field method for solving incompressible two-phase flows.* Journal of Computational Physics, 230, 185–204.

---

## Requirements

- Python 3.9+
- NumPy
- Matplotlib
