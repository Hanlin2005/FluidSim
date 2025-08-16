"""
JAX Finite-Volume CFD Simulator (2D Incompressible, MAC Grid)
-----------------------------------------------------------------
Educational reference implementation of a 2D incompressible Navier–Stokes
solver using the finite-volume method on a staggered (MAC) grid.

Highlights:
- FV control volumes: u on vertical face centers, v on horizontal face centers,
  pressure p at cell centers.
- Semi-implicit projection method: advance viscous+advective terms, then solve a
  pressure Poisson equation to enforce incompressibility, and finally correct the
  velocities.
- Upwind advection, central diffusion.
- Simple Jacobi pressure solver (GPU/TPU-friendly; swap in better solvers later).
- JAX jit/vmap/lax.scan friendly; functional style state updates.

This code is compact enough to read end-to-end, with each method clearly
annotated to explain *what it does* in an FV context.

NOTE: This prioritizes clarity over maximum performance. For serious use, you
would want: multigrid/PCG for pressure, TVD/WENO advection, better boundary
handling, and adaptive time stepping.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import lax, tree_util

Array = jnp.ndarray

# -----------------------------------------------------------------------------
# Grid + state containers
# -----------------------------------------------------------------------------

@dataclass
class Grid:
    """Uniform Cartesian grid describing the FV control volumes.

    Attributes
    ----------
    nx, ny : int
        Number of *pressure cells* in x and y. Velocity unknowns live on faces.
    Lx, Ly : float
        Physical domain size (meters).
    dx, dy : float
        Cell size.
    periodic : bool
        If True, apply periodic BCs for all fields; otherwise use no-slip walls
        (u=v=0 at domain boundary) and zero-normal-gradient for pressure.
    """
    nx: int
    ny: int
    Lx: float
    Ly: float
    periodic: bool = False

    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny

@dataclass
class FlowState:
    """Holds velocity and pressure fields on a MAC grid.

    Shapes
    ------
    - u: (ny, nx+1) -> x-velocity on vertical faces of pressure cells
    - v: (ny+1, nx) -> y-velocity on horizontal faces of pressure cells
    - p: (ny, nx)   -> cell-centered pressure
    """
    u: Array
    v: Array
    p: Array

# Register FlowState as a JAX PyTree so it can be passed to jit/scan
# (tells JAX how to flatten/unflatten the dataclass)

def _flowstate_flatten(fs: FlowState):
    children = (fs.u, fs.v, fs.p)
    aux = None
    return children, aux


def _flowstate_unflatten(aux, children):
    u, v, p = children
    return FlowState(u=u, v=v, p=p)

# Register with JAX once at import time
try:
    tree_util.register_pytree_node(FlowState, _flowstate_flatten, _flowstate_unflatten)
except Exception:
    # Safe to ignore if this module gets reloaded in an interactive session
    pass

@dataclass
class FluidParams:
    rho: float  # density
    nu: float   # kinematic viscosity (ν = μ/ρ)
    dt: float   # time step

# -----------------------------------------------------------------------------
# Utility: boundary conditions
# -----------------------------------------------------------------------------

def apply_bc_u(u: Array, grid: Grid) -> Array:
    """Apply boundary conditions to face-centered u (x-velocity).

    If periodic: wrap ghost faces.
    Else: enforce no-slip at walls (u=0 on left/right boundaries) and copy
    interior values at top/bottom for zero normal flow through walls.
    """
    if grid.periodic:
        # Wrap in x; wrap in y by copying rows
        u = u.at[:, 0].set(u[:, -2])
        u = u.at[:, -1].set(u[:, 1])
        u = u.at[0, :].set(u[-2, :])
        u = u.at[-1, :].set(u[1, :])
        return u
    # No-slip walls: u=0 at x-boundaries
    u = u.at[:, 0].set(0.0)
    u = u.at[:, -1].set(0.0)
    # Impermeable top/bottom: copy interior (Neumann in y)
    u = u.at[0, :].set(u[1, :])
    u = u.at[-1, :].set(u[-2, :])
    return u


def apply_bc_v(v: Array, grid: Grid) -> Array:
    """Apply boundary conditions to face-centered v (y-velocity)."""
    if grid.periodic:
        v = v.at[:, 0].set(v[:, -2])
        v = v.at[:, -1].set(v[:, 1])
        v = v.at[0, :].set(v[-2, :])
        v = v.at[-1, :].set(v[1, :])
        return v
    # No-slip walls: v=0 at y-boundaries
    v = v.at[0, :].set(0.0)
    v = v.at[-1, :].set(0.0)
    # Impermeable left/right: copy interior (Neumann in x)
    v = v.at[:, 0].set(v[:, 1])
    v = v.at[:, -1].set(v[:, -2])
    return v


def apply_bc_p(p: Array, grid: Grid) -> Array:
    """Apply boundary conditions to cell-centered pressure.

    For periodic: wrap. Otherwise use zero-normal-gradient at walls (copy).
    """
    if grid.periodic:
        p = p.at[:, 0].set(p[:, -2])
        p = p.at[:, -1].set(p[:, 1])
        p = p.at[0, :].set(p[-2, :])
        p = p.at[-1, :].set(p[1, :])
        return p
    p = p.at[:, 0].set(p[:, 1])
    p = p.at[:, -1].set(p[:, -2])
    p = p.at[0, :].set(p[1, :])
    p = p.at[-1, :].set(p[-2, :])
    return p

# -----------------------------------------------------------------------------
# Finite-volume differential operators on MAC grid
# -----------------------------------------------------------------------------

def divergence(u: Array, v: Array, grid: Grid) -> Array:
    """Divergence at cell centers: (∂u/∂x + ∂v/∂y).

    FV interpretation: net outflow across cell faces / cell volume.
    Inputs are face-centered velocities. Output shape is (ny, nx).
    """
    dx, dy = grid.dx, grid.dy
    dudx = (u[:, 1:] - u[:, :-1]) / dx  # (ny, nx)
    dvdy = (v[1:, :] - v[:-1, :]) / dy  # (ny, nx)
    return dudx + dvdy


def grad_p(u: Array, v: Array, p: Array, grid: Grid) -> Tuple[Array, Array]:
    """Pressure gradient projected to u/v face locations.

    Returns (dpdx_on_u_faces, dpdy_on_v_faces) with shapes like u and v.
    """
    dx, dy = grid.dx, grid.dy
    # Interpolate cell-centered p to u faces in x, then take centered diff
    # For u at face (i+1/2,j), grad p ≈ (p[i+1,j] - p[i,j]) / dx
    dpdx_u = (p[:, 1:] - p[:, :-1]) / dx
    # pad to match u shape (ny, nx+1)
    dpdx_u = jnp.pad(dpdx_u, ((0, 0), (1, 0)))

    # For v at face (i,j+1/2), grad p ≈ (p[i,j+1] - p[i,j]) / dy
    dpdy_v = (p[1:, :] - p[:-1, :]) / dy
    # pad to match v shape (ny+1, nx)
    dpdy_v = jnp.pad(dpdy_v, ((1, 0), (0, 0)))
    return dpdx_u, dpdy_v


def laplacian_u(u: Array, grid: Grid) -> Array:
    """Laplacian of u on u-faces using 5-point stencil (FV/FD equivalent)."""
    dx2, dy2 = grid.dx**2, grid.dy**2
    u_e = u[:, 2:]
    u_w = u[:, :-2]
    u_n = u[2:, 1:-1]
    u_s = u[:-2, 1:-1]
    u_c = u[1:-1, 1:-1]
    lap = (u_e + u_w - 2.0 * u_c) / dx2 + (u_n + u_s - 2.0 * u_c) / dy2
    return jnp.pad(lap, ((1, 1), (1, 1)))


def laplacian_v(v: Array, grid: Grid) -> Array:
    """Laplacian of v on v-faces using 5-point stencil (FV/FD equivalent)."""
    dx2, dy2 = grid.dx**2, grid.dy**2
    v_e = v[:, 2:]
    v_w = v[:, :-2]
    v_n = v[2:, 1:-1]
    v_s = v[:-2, 1:-1]
    v_c = v[1:-1, 1:-1]
    lap = (v_e + v_w - 2.0 * v_c) / dx2 + (v_n + v_s - 2.0 * v_c) / dy2
    return jnp.pad(lap, ((1, 1), (1, 1)))

# -----------------------------------------------------------------------------
# Convective (advection) fluxes (first-order upwind)
# -----------------------------------------------------------------------------

def advect_u(u: Array, v: Array, grid: Grid) -> Array:
    """Upwind advection of u stored on vertical faces.

    Computes (u · ∇)u at u-face locations. We reconstruct donor-cell fluxes using
    face-normal velocities and interpolated transverse velocities.
    """
    dx, dy = grid.dx, grid.dy
    # Interpolate velocities to u-face-centered stencil points
    u_c = u  # already on u faces
    # u at neighboring faces in x for upwinding
    u_e = jnp.roll(u_c, -1, axis=1)
    u_w = jnp.roll(u_c, 1, axis=1)

    # Interpolate v to u-face centers (average the two adjacent v faces)
    v_on_u = 0.5 * (v[:-1, :] + v[1:, :])
    v_on_u = 0.5 * (v_on_u[:, 1:] + v_on_u[:, :-1])  # to (ny, nx+1)

    # Upwind in x
    flux_x = jnp.where(u_c >= 0.0, u_c * (u_c - u_w) / dx, u_c * (u_e - u_c) / dx)
    # Upwind in y (use v at u faces)
    u_n = jnp.roll(u_c, -1, axis=0)
    u_s = jnp.roll(u_c, 1, axis=0)
    v_c = v_on_u
    flux_y = jnp.where(v_c >= 0.0, v_c * (u_c - u_s) / dy, v_c * (u_n - u_c) / dy)

    return flux_x + flux_y


def advect_v(u: Array, v: Array, grid: Grid) -> Array:
    """Upwind advection of v stored on horizontal faces: (u · ∇)v."""
    dx, dy = grid.dx, grid.dy
    v_c = v
    v_n = jnp.roll(v_c, -1, axis=0)
    v_s = jnp.roll(v_c, 1, axis=0)

    # Interpolate u to v-face centers
    u_on_v = 0.5 * (u[:, :-1] + u[:, 1:])
    u_on_v = 0.5 * (u_on_v[1:, :] + u_on_v[:-1, :])  # to (ny+1, nx)

    # Upwind in y
    flux_y = jnp.where(v_c >= 0.0, v_c * (v_c - v_s) / dy, v_c * (v_n - v_c) / dy)
    # Upwind in x (use u at v faces)
    v_e = jnp.roll(v_c, -1, axis=1)
    v_w = jnp.roll(v_c, 1, axis=1)
    u_c = u_on_v
    flux_x = jnp.where(u_c >= 0.0, u_c * (v_c - v_w) / dx, u_c * (v_e - v_c) / dx)

    return flux_x + flux_y

# -----------------------------------------------------------------------------
# Pressure Poisson solver (Jacobi iterations)
# -----------------------------------------------------------------------------

def jacobi_pressure_solve(rhs: Array, grid: Grid, iters: int = 200, omega: float = 1.0) -> Array:
    """Solve ∇²p = rhs with simple Jacobi iterations.

    Parameters
    ----------
    rhs : (ny, nx) Array
        Right-hand side = (ρ/Δt) * div(u*) ; u* is intermediate velocity.
    iters : int
        Number of Jacobi sweeps (increase for tighter divergence).
    omega : float
        Relaxation factor (1.0 = Jacobi; (0,2) for weighted Jacobi).
    """
    dx2, dy2 = grid.dx**2, grid.dy**2
    inv_denom = 1.0 / (2.0 * (1.0/dx2 + 1.0/dy2))

    def body_fun(p):
        p = apply_bc_p(p, grid)
        p_e = jnp.pad(p[:, 1:], ((0, 0), (0, 1)))
        p_w = jnp.pad(p[:, :-1], ((0, 0), (1, 0)))
        p_n = jnp.pad(p[1:, :], ((0, 1), (0, 0)))
        p_s = jnp.pad(p[:-1, :], ((1, 0), (0, 0)))
        p_new = ((p_e + p_w) / dx2 + (p_n + p_s) / dy2 - rhs) * inv_denom
        return (1 - omega) * p + omega * p_new

    def scan_fun(p, _):
        p = body_fun(p)
        return p, None

    p0 = jnp.zeros_like(rhs)
    p_final, _ = lax.scan(scan_fun, p0, None, length=iters)
    return apply_bc_p(p_final, grid)

# -----------------------------------------------------------------------------
# Time stepping: projection method
# -----------------------------------------------------------------------------

def time_step(state: FlowState, params: FluidParams, grid: Grid) -> FlowState:
    """Advance one time step using a projection method.

    Steps
    -----
    1) Apply BCs to u, v.
    2) Build RHS for momentum using upwind advection and viscous diffusion.
    3) Compute provisional velocities u*, v* (explicit Euler for advection;
       semi-implicit for diffusion via simple explicit Laplacian here for clarity).
    4) Solve Poisson for pressure from div(u*).
    5) Correct velocities: u^{n+1} = u* - (Δt/ρ) ∂p/∂x, v^{n+1} similarly.
    """
    rho, nu, dt = params.rho, params.nu, params.dt

    u = apply_bc_u(state.u, grid)
    v = apply_bc_v(state.v, grid)

    # Advection and diffusion terms (on faces)
    adv_u = advect_u(u, v, grid)
    adv_v = advect_v(u, v, grid)
    diff_u = nu * laplacian_u(u, grid)
    diff_v = nu * laplacian_v(v, grid)

    # Provisional velocities u*, v*
    u_star = u + dt * (-adv_u + diff_u)
    v_star = v + dt * (-adv_v + diff_v)

    u_star = apply_bc_u(u_star, grid)
    v_star = apply_bc_v(v_star, grid)

    # Pressure from divergence of u*
    div_star = divergence(u_star, v_star, grid)
    rhs = (rho / dt) * div_star
    p_new = jacobi_pressure_solve(rhs, grid)

    # Correct velocities to make them divergence-free
    dpdx_u, dpdy_v = grad_p(u_star, v_star, p_new, grid)
    u_np1 = u_star - (dt / rho) * dpdx_u
    v_np1 = v_star - (dt / rho) * dpdy_v

    u_np1 = apply_bc_u(u_np1, grid)
    v_np1 = apply_bc_v(v_np1, grid)

    return FlowState(u=u_np1, v=v_np1, p=p_new)

# -----------------------------------------------------------------------------
# Simulation driver + helpers
# -----------------------------------------------------------------------------

def cfl_number(state: FlowState, grid: Grid, dt: float, nu: float) -> float:
    """Compute a simple CFL-like stability estimate (convective + diffusive)."""
    umax = jnp.max(jnp.abs(state.u))
    vmax = jnp.max(jnp.abs(state.v))
    conv = jnp.maximum(umax * dt / grid.dx, vmax * dt / grid.dy)
    diff = nu * dt * (1.0 / grid.dx**2 + 1.0 / grid.dy**2)
    return float(jnp.maximum(conv, diff))


def simulate(state0: FlowState, params: FluidParams, grid: Grid, steps: int) -> FlowState:
    """Run `steps` time steps using `lax.scan` for performance."""
    def step_fn(state, _):
        new_state = time_step(state, params, grid)
        return new_state, None

    final_state, _ = lax.scan(step_fn, state0, None, length=steps)
    return final_state

# -----------------------------------------------------------------------------
# Problem setup: lid-driven cavity (classic FV benchmark)
# -----------------------------------------------------------------------------

def lid_driven_cavity_ic(grid: Grid, lid_velocity: float = 1.0) -> FlowState:
    """Initialize a quiescent cavity with a moving top lid (u=U_lid on top)."""
    ny, nx = grid.ny, grid.nx
    u = jnp.zeros((ny, nx + 1))
    v = jnp.zeros((ny + 1, nx))
    p = jnp.zeros((ny, nx))
    # Impose lid velocity on top u faces (between top cells)
    u = u.at[-1, :].set(lid_velocity)
    return FlowState(u=u, v=v, p=p)

# -----------------------------------------------------------------------------
# Main: minimal example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # Grid and fluid
    grid = Grid(nx=64, ny=64, Lx=1.0, Ly=1.0, periodic=False)
    params = FluidParams(rho=1.0, nu=1e-3, dt=1e-3)

    # Initial condition: lid-driven cavity
    state = lid_driven_cavity_ic(grid, lid_velocity=1.0)

    # JIT the timestepper once (warm-up)
    time_step_jit = jax.jit(lambda s: time_step(s, params, grid))
    state = time_step_jit(state)

    # Run a short simulation
    steps = 500
    sim_fn = jax.jit(lambda s: simulate(s, params, grid, steps))
    state = sim_fn(state)

    # Report divergence as a sanity check
    div = divergence(state.u, state.v, grid)
    max_div = float(jnp.max(jnp.abs(div)))
    print(f"Max divergence after {steps} steps: {max_div:.3e}")
    print(f"CFL estimate: {cfl_number(state, grid, params.dt, params.nu):.3f}")

    # NOTE: For visualization, export `state.u`, `state.v` (interpolate to cell
    # centers), and `state.p` to your plotting tool of choice.
