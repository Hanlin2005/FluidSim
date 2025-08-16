# Differentiable 2D CFD in JAX (incompressible, periodic, vorticity–streamfunction form)
# - Pseudo-spectral with FFTs
# - AB2 for advection, CN for diffusion
# - Fully differentiable wrt init conds, viscosity, forcing params, etc.

from dataclasses import dataclass
from typing import Callable, Tuple
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.experimental import mesh_utils
from jax import config
config.update("jax_enable_x64", True)

Array = jnp.ndarray

def default_forcing(x: Array, y: Array, t: float, params) -> Tuple[Array, Array]:
    """
    Body force field f = (fx, fy) on the velocity. Periodic Taylor-Green-style drive.
    Params can contain amplitude 'A' and wavenumber 'k'.
    """
    A = params.get("A", 0.0)
    k = params.get("k", 2.0 * jnp.pi)  # one oscillation over domain length if L=1
    fx = A * jnp.sin(k * x) * jnp.cos(k * y)
    fy = -A * jnp.cos(k * x) * jnp.sin(k * y)
    return fx, fy

@dataclass
class SpectralOperators:
    N: int
    L: float
    kx: Array
    ky: Array
    k2: Array
    ikx: Array
    iky: Array
    lap: Array
    inv_lap: Array
    dealias: Array

def make_spectral_ops(N: int, L: float, dealias: bool = True) -> SpectralOperators:
    # Wavenumbers (FFT ordering)
    k = jnp.fft.fftfreq(N, d=L/N) * 2.0 * jnp.pi  # shape (N,)
    kx = jnp.repeat(k[:, None], N, axis=1)
    ky = jnp.repeat(k[None, :], N, axis=0)
    k2 = kx**2 + ky**2
    ikx = 1j * kx
    iky = 1j * ky
    lap = -k2
    inv_lap = jnp.where(k2 == 0.0, 0.0, -1.0 / k2)  # inverse Laplacian with zero-mean
    if dealias:
        # 2/3-rule dealiasing mask
        kcut = (2.0/3.0) * (N//2) * 2.0 * jnp.pi / (L)
        mask = (jnp.abs(kx) <= kcut) & (jnp.abs(ky) <= kcut)
        dealias_mask = mask.astype(jnp.float64)
    else:
        dealias_mask = jnp.ones((N, N), dtype=jnp.float64)
    return SpectralOperators(N, L, kx, ky, k2, ikx, iky, lap, inv_lap, dealias_mask)

@jax.tree_util.register_pytree_node_class
@dataclass
class CFDState:
    # make all fields JAX arrays so they’re valid leaves
    t: jnp.ndarray
    w_hat: jnp.ndarray
    N_hat_prev: jnp.ndarray
    is_first: jnp.ndarray  # boolean scalar as array

    # pytree plumbing
    def tree_flatten(self):
        return (self.t, self.w_hat, self.N_hat_prev, self.is_first), None
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        t, w_hat, N_hat_prev, is_first = children
        return cls(t, w_hat, N_hat_prev, is_first)

@dataclass
class CFDSystem:
    ops: SpectralOperators
    nu: float
    forcing: Callable[[Array, Array, float, dict], Tuple[Array, Array]]
    forcing_params: dict

def curl_of_force_hat(ops: SpectralOperators, fx: Array, fy: Array) -> Array:
    # Return z-component of curl(f) in spectral space: (∂fy/∂x - ∂fx/∂y)
    fx_hat = jnp.fft.fft2(fx)
    fy_hat = jnp.fft.fft2(fy)
    return ops.ikx * fy_hat - ops.iky * fx_hat

def velocity_from_vorticity_hat(ops: SpectralOperators, w_hat: Array) -> Tuple[Array, Array]:
    # Streamfunction ψ solves ∇²ψ = -ω  => ψ_hat = inv_lap * (-ω_hat) = +inv_lap * (-1) * ω_hat
    psi_hat = ops.inv_lap * (-w_hat)
    # u = (∂ψ/∂y, -∂ψ/∂x) => in Fourier: u_hat = (iky*psi_hat, -ikx*psi_hat)
    ux_hat = ops.iky * psi_hat
    uy_hat = -ops.ikx * psi_hat
    ux = jnp.fft.ifft2(ux_hat).real
    uy = jnp.fft.ifft2(uy_hat).real
    return ux, uy

def nonlinear_term_hat(ops: SpectralOperators, w_hat: Array) -> Array:
    # Compute N = u · ∇ω in physical space, then FFT back, with dealiasing
    w = jnp.fft.ifft2(w_hat).real
    ux, uy = velocity_from_vorticity_hat(ops, w_hat)
    # Gradients of vorticity via spectral are also fine; to keep symmetry we’ll do spectral grads:
    wx_hat = ops.ikx * jnp.fft.fft2(w)
    wy_hat = ops.iky * jnp.fft.fft2(w)
    wx = jnp.fft.ifft2(wx_hat).real
    wy = jnp.fft.ifft2(wy_hat).real
    N = ux * wx + uy * wy
    N_hat = jnp.fft.fft2(N) * ops.dealias
    return N_hat
def step(system: CFDSystem, state: CFDState, dt: float) -> CFDState:
    ops = system.ops
    nu = system.nu

    N_hat_n = nonlinear_term_hat(ops, state.w_hat)
    N_hat_ab = jnp.where(
        state.is_first,              # first step: AB1 (just N^n)
        N_hat_n,
        1.5 * N_hat_n - 0.5 * state.N_hat_prev  # AB2 thereafter
    )

    # Forcing at time t
    N = ops.N
    L = ops.L
    x = jnp.linspace(0.0, L, N, endpoint=False)
    y = jnp.linspace(0.0, L, N, endpoint=False)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    fx, fy = system.forcing(X, Y, state.t, system.forcing_params)
    curlf_hat = curl_of_force_hat(ops, fx, fy)

    # CN diffusion
    a = 1.0 - dt * nu * ops.lap / 2.0
    b = 1.0 + dt * nu * ops.lap / 2.0
    rhs = b * state.w_hat - dt * (N_hat_ab - curlf_hat)
    w_hat_np1 = rhs / a

    # flip is_first -> False after the first step
    return CFDState(
        t=state.t + dt,
        w_hat=w_hat_np1,
        N_hat_prev=N_hat_n,
        is_first=jnp.array(False)
    )


def rollout(system: CFDSystem, state: CFDState, dt: float, steps: int) -> CFDState:
    def body(s, _):
        s_next = step(system, s, dt)
        return s_next, None
    state, _ = jax.lax.scan(body, state, xs=None, length=steps)
    return state

def kinetic_energy(ops: SpectralOperators, w_hat: Array) -> float:
    ux, uy = velocity_from_vorticity_hat(ops, w_hat)
    return 0.5 * jnp.mean(ux**2 + uy**2)

def divergence(ops: SpectralOperators, w_hat: Array) -> float:
    # Should be ~0 numerically
    ux, uy = velocity_from_vorticity_hat(ops, w_hat)
    ux_hat = jnp.fft.fft2(ux)
    uy_hat = jnp.fft.fft2(uy)
    div_hat = ops.ikx * ux_hat + ops.iky * uy_hat
    div = jnp.fft.ifft2(div_hat).real
    return jnp.linalg.norm(div) / div.size

# -------------------------------
# Example usage + differentiability
# -------------------------------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    N = 128
    L = 1.0
    dt = 5e-4
    steps = 2000
    nu = 1e-3

    ops = make_spectral_ops(N=N, L=L, dealias=True)
    ops.N = N  # small convenience so we can access inside jit'ed step
    ops.L = L

    # Initial vorticity: small random divergence-free flow via random streamfunction
    psi0 = 1e-3 * jax.random.normal(key, (N, N))
    psi0_hat = jnp.fft.fft2(psi0)
    w0_hat = -ops.lap * psi0_hat  # ω = -∇²ψ

    system = CFDSystem(
        ops=ops,
        nu=nu,
        forcing=default_forcing,
        forcing_params={"A": 5e-2, "k": 2.0 * jnp.pi}
    )

    state0 = CFDState(
        t=jnp.array(0.0),
        w_hat=w0_hat,
        N_hat_prev=jnp.zeros_like(w0_hat),  # no None
        is_first=jnp.array(True)
    )

    # Roll out
    final_state = rollout(system, state0, dt=dt, steps=steps)

    # Compute a loss: match target KE at final time and penalize divergence
    target_ke = 0.002
    loss = (kinetic_energy(ops, final_state.w_hat) - target_ke) ** 2 + 1e-4 * divergence(ops, final_state.w_hat)

    print("Final time:", final_state.t)
    print("Final KE:", float(kinetic_energy(ops, final_state.w_hat)))
    print("Divergence L2 per DOF:", float(divergence(ops, final_state.w_hat)))
    print("Loss:", float(loss))

    # --------- Differentiation example ----------
    # Gradient wrt forcing amplitude A
    def loss_wrt_A(A: float) -> float:
        sysA = CFDSystem(
            ops=ops, nu=nu, forcing=default_forcing, forcing_params={"A": A, "k": 2.0 * jnp.pi}
        )
        stA = rollout(sysA, state0, dt=dt, steps=steps)
        return (kinetic_energy(ops, stA.w_hat) - target_ke) ** 2 + 1e-4 * divergence(ops, stA.w_hat)

    dL_dA = grad(loss_wrt_A)(system.forcing_params["A"])
    print("dLoss/dA:", float(dL_dA))

    # Gradient wrt viscosity (useful for ID/estimation)
    def loss_wrt_nu(nu_val: float) -> float:
        sysN = CFDSystem(ops=ops, nu=nu_val, forcing=default_forcing, forcing_params=system.forcing_params)
        stN = rollout(sysN, state0, dt=dt, steps=steps)
        return (kinetic_energy(ops, stN.w_hat) - target_ke) ** 2

    dL_dnu = grad(loss_wrt_nu)(nu)
    print("dLoss/dnu:", float(dL_dnu))
