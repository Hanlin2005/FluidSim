# Physics-Informed Neural Networks for Incompressible Viscous Flow

A research implementation of Physics-Informed Neural Networks (PINNs) applied to the 2D lid-driven cavity benchmark, with differentiable solvers, error-guided importance sampling, and transfer learning across flow regimes.

---

## Overview

This project investigates the use of PINNs as mesh-free surrogates for incompressible Navier–Stokes flow. The lid-driven cavity problem — a standard CFD benchmark with a known, non-trivial steady-state solution — serves as the primary test case. Three complementary research directions are pursued:

1. **Reference solver**: A finite-difference Navier–Stokes solver (NumPy) provides ground-truth velocity and pressure fields used for training supervision and error evaluation.
2. **PINN training**: A fully-connected neural network is trained to satisfy the steady incompressible Navier–Stokes equations via a PDE-residual loss on the domain interior and a data-matching loss on boundaries.
3. **Adaptive sampling**: Importance sampling strategies concentrate collocation points in high-error regions, accelerating convergence compared to uniform random sampling.
4. **Transfer learning**: A pre-trained PINN is fine-tuned to a different lid velocity (flow regime) with partial weight freezing, demonstrating the generalization capacity of the learned representation.
5. **Differentiable simulation**: JAX-based implementations of the FDM and a pseudo-spectral solver support end-to-end differentiation through the time-stepping loop, enabling gradient-based parameter identification (e.g., sensitivity of kinetic energy to lid speed).

---

## Repository Structure

```
FluidSim/
├── fdm/                              # Finite-difference reference solvers
│   ├── lid_cavity_FDM.py             # NumPy solver: pressure-Poisson + velocity update
│   └── lid_cavity_FDM_differentiable.py  # JAX solver: differentiable w.r.t. boundary conditions
│
├── pinn/                             # Physics-Informed Neural Network experiments
│   ├── lid_cavity_PINN.py            # Network architecture, PDE loss, training, evaluation
│   ├── lid_cavity_sampling.py        # Error-guided importance sampling for collocation
│   ├── lid_cavity_transfer.py        # Transfer learning to a new flow regime
│   └── lid_cavity_loss_evolution.py  # Training diagnostics: loss and L2 error over epochs
│
├── spectral/                         # Higher-order differentiable solvers (JAX)
│   ├── CFD.py                        # Pseudo-spectral solver (vorticity–streamfunction, AB2/CN)
│   └── diff_sim.py                   # Finite-volume solver on staggered MAC grid
│
├── models/                           # Saved model checkpoints
│   └── lid_pinn_full.pth             # Pre-trained PINN weights (U_lid = 1.0, Re = 20)
│
└── plots/                            # Generated figures
    ├── lid_streamlines.png           # FDM steady-state streamlines
    ├── lid_vorticity.png             # Vorticity field
    ├── mse_loss_lid_cavity.png       # Training loss curve
    └── PINN-transfer.png             # Transfer learning comparison
```

---

## Methods

### 1. Finite-Difference Navier–Stokes Solver (`fdm/`)

The incompressible Navier–Stokes equations are discretized on a uniform staggered grid using second-order central differences. Pressure is obtained by solving a Poisson equation via Jacobi iteration, and velocity is advanced with a fractional-step method. The domain is a 2×2 cavity with a moving top lid at velocity *U*.

**Governing equations** (steady-state, dimensionless):

$$u \cdot \nabla u = -\frac{1}{\rho}\nabla p + \nu \nabla^2 u, \qquad \nabla \cdot u = 0$$

**Parameters**: 41×41 grid, ν = 0.1, ρ = 1, U = 1.0 (Re = 20).

The JAX variant (`lid_cavity_FDM_differentiable.py`) wraps the time loop in `jax.lax.scan` and differentiates through all 500 time steps via reverse-mode AD, yielding ∂KE/∂U analytically.

### 2. Physics-Informed Neural Network (`pinn/lid_cavity_PINN.py`)

**Architecture**: 9 hidden layers × 20 neurons, Tanh activations, input (x, y) → output (u, v, p).

**Loss function**:

$$\mathcal{L} = \underbrace{\mathcal{L}_{\text{PDE}}}_{\text{NS residuals on interior}} + \underbrace{\mathcal{L}_{\text{BC}}}_{\text{supervised boundary data}}$$

The PDE residual is computed via automatic differentiation (PyTorch `autograd`) of the network output with respect to input coordinates, enforcing:
- *x*-momentum, *y*-momentum (Navier–Stokes)
- Continuity (divergence-free condition)

Boundary conditions are enforced by minimizing MSE against FDM-interpolated values at sampled boundary points.

**Optimizer**: Adam, lr = 1×10⁻³, 5000 epochs.

**Evaluation**: Relative L2 error over the full grid against the FDM reference:

$$\varepsilon = \frac{\|\hat{q} - q_{\text{FDM}}\|_2}{\|q_{\text{FDM}}\|_2 + \epsilon}$$

### 3. Importance Sampling (`pinn/lid_cavity_sampling.py`)

Uniform random collocation can be inefficient when errors are spatially non-uniform. Two error-guided samplers are implemented:

- **`sample_points_by_error`**: Builds a probability distribution over interior grid points proportional to the pointwise relative error (with temperature scaling and a tunable uniform fraction for exploration).
- **`sample_boundary_by_error`**: Samples boundary strips weighted by local error, concentrating capacity near poorly-fit boundary regions.

Both samplers support replacement and are JAX-agnostic (operate on NumPy arrays).

### 4. Transfer Learning (`pinn/lid_cavity_transfer.py`)

A PINN pre-trained at U = 1.0 (Re = 20) is fine-tuned to U = 0.01 (Re = 0.2, near-Stokes regime) by:

1. **Partial freezing**: The first *k* linear layers are frozen; only the remaining layers are updated (k = 4 by default).
2. **Error-guided sampling**: Collocation points are drawn by importance sampling on the current error map, not uniformly.
3. **Combined loss**: PDE residual + boundary data + sparse interior supervision.

This demonstrates that features learned at one Reynolds number transfer usefully to another, requiring far fewer epochs than training from scratch.

**Training diagnostics** (`pinn/lid_cavity_loss_evolution.py`) tracks both the training loss and the relative L2 error at every epoch, producing convergence curves on a log scale.

---

## Results

| Experiment | Key Result |
|---|---|
| FDM reference solver | Converged lid-cavity flow at Re = 20 in 500 steps |
| PINN (PDE + BC loss) | Relative L2 error < threshold after 5000 epochs |
| Importance sampling | Reduced collocation budget vs. uniform sampling for equal accuracy |
| Transfer learning | PINN adapts to U = 0.01 regime in 500 fine-tuning epochs |
| Differentiable FDM | ∂KE/∂U computed analytically through 500 time steps via JAX AD |

---

## Installation

```bash
git clone https://github.com/Hanlin2005/FluidSim.git
cd FluidSim
pip install -r requirements.txt
```

**Dependencies**:

| Package | | Purpose |
|---|---|
| `numpy` | FDM solver, array operations |
| `scipy` | Bilinear interpolation (`RegularGridInterpolator`) |
| `torch` | PINN training, automatic differentiation |
| `matplotlib` | Visualization |
| `jax[cpu]` / `jax[cuda]` | Differentiable solvers, JIT compilation |

---

## Usage

All scripts are run as Python modules from the project root so that inter-package imports resolve correctly.

**Run the FDM reference solver:**
```bash
python -m fdm.lid_cavity_FDM
```

**Train the PINN from scratch:**
```bash
python -m pinn.lid_cavity_PINN
```
Saves a checkpoint to `models/lid_pinn_full.pth`.

**Run transfer learning experiment:**
```bash
python -m pinn.lid_cavity_transfer
```
Loads `models/lid_pinn_full.pth` and fine-tunes to U = 0.01.

**Track loss/error convergence:**
```bash
python -m pinn.lid_cavity_loss_evolution
```

**Differentiable FDM (JAX):**
```bash
python -m fdm.lid_cavity_FDM_differentiable
```

---

## Technical Notes

- The PINN uses the FDM solution only as supervision signal on boundary/interior points — it does not see the FDM grid structure during training. Interior physics are enforced purely through PDE residuals.
- The JAX FDM differentiates through 500×50 Jacobi iterations (25,000 total linear solves) via reverse-mode AD. Memory scales with the number of time steps; use `jax.checkpoint` for longer rollouts.
- All random seeds are not fixed by default; results will vary across runs. Set `torch.manual_seed` / `jax.random.PRNGKey` for reproducibility.

---