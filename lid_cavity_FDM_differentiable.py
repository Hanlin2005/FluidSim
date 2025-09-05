#THIS CODE IS FOR THE LID CAVITY METHOD USING FDM IMPLEMENTED FOR DIFFERENTIABILITY
import jax
import jax.numpy as jnp
from jax import lax

# domain
Lx, Ly = 2.0, 2.0
nx, ny = 41, 41
x = jnp.linspace(0.0, Lx, nx)
y = jnp.linspace(0.0, Ly, ny)
dx = Lx / (nx - 1)
dy = Ly / (ny - 1)

rho = 1.0
nu  = 0.1
dt  = 1e-3
nit = 50
nt  = 500
U_lid = 1.0

def build_b(u, v, dx, dy, dt, rho):
    dudx = (u[:, 2:] - u[:, :-2])/(2*dx)
    dvdy = (v[2:, :] - v[:-2, :])/(2*dy)
    du_dy = (u[2:, 1:-1] - u[:-2, 1:-1])/(2*dy)
    dv_dx = (v[1:-1, 2:] - v[1:-1, :-2])/(2*dx)
    term = (1/dt)*(dudx[1:-1, :] + dvdy[:, 1:-1])
    b = jnp.zeros_like(u)
    center = (rho * (term
                     - dudx[1:-1, :]**2
                     - 2*du_dy*dv_dx
                     - dvdy[:, 1:-1]**2))
    b = b.at[1:-1,1:-1].set(center)
    return b

def apply_pressure_bcs(p):
    p = p.at[:, -1].set(p[:, -2])  # dp/dx=0 at x=Lx
    p = p.at[0, :].set(p[1, :])    # dp/dy=0 at y=0
    p = p.at[:, 0].set(p[:, 1])    # dp/dx=0 at x=0
    p = p.at[-1, :].set(0.0)       # p=0 at y=Ly
    return p

def poisson_jacobi(p0, b, dx, dy, nit):
    def body(_, state):
        p = state
        pn = p
        p_new = (((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                  (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2) /
                 (2 * (dx**2 + dy**2))
                 - dx**2 * dy**2 * b[1:-1,1:-1] / (2 * (dx**2 + dy**2)))
        p = p.at[1:-1,1:-1].set(p_new)
        p = apply_pressure_bcs(p)
        return p
    p = apply_pressure_bcs(p0)
    p = lax.fori_loop(0, nit, body, p)
    return p

def apply_velocity_bcs(u, v, U):
    # cavity: no-slip on walls, moving lid at top
    u = u.at[0, :].set(0.0)    # y=0
    u = u.at[-1, :].set(U)     # y=Ly (lid)
    u = u.at[:, 0].set(0.0)
    u = u.at[:, -1].set(0.0)

    v = v.at[0, :].set(0.0)
    v = v.at[-1, :].set(0.0)
    v = v.at[:, 0].set(0.0)
    v = v.at[:, -1].set(0.0)
    return u, v

def step(state, params):
    u, v, p = state
    dx, dy, dt, rho, nu, U = params

    b = build_b(u, v, dx, dy, dt, rho)
    p = poisson_jacobi(p, b, dx, dy, nit)

    un, vn = u, v

    u_center = (un[1:-1,1:-1] 
                - un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,0:-2])
                - vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[0:-2,1:-1])
                - dt/(2*rho*dx)*(p[1:-1,2:] - p[1:-1,0:-2])
                + nu*dt*((un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,0:-2])/dx**2
                       + (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[0:-2,1:-1])/dy**2))

    v_center = (vn[1:-1,1:-1]
                - un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,0:-2])
                - vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[0:-2,1:-1])
                - dt/(2*rho*dy)*(p[2:,1:-1] - p[0:-2,1:-1])
                + nu*dt*((vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,0:-2])/dx**2
                       + (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[0:-2,1:-1])/dy**2))

    u = u.at[1:-1,1:-1].set(u_center)
    v = v.at[1:-1,1:-1].set(v_center)
    u, v = apply_velocity_bcs(u, v, U)

    return (u, v, p), None

@jax.jit
def simulate(u0, v0, p0, params):
    init = (u0, v0, p0)
    (uT, vT, pT), _ = lax.scan(lambda s, _: step(s, params), init, xs=None, length=nt)
    return uT, vT, pT

# Bilinear sampler for (xq, yq) in domain (regular grid)
def bilinear_sample(field, xq, yq, dx, dy):
    ix = jnp.clip(jnp.floor(xq/dx).astype(int), 0, field.shape[1]-2)
    iy = jnp.clip(jnp.floor(yq/dy).astype(int), 0, field.shape[0]-2)
    fx = xq/dx - ix
    fy = yq/dy - iy
    f00 = field[iy,   ix  ]
    f10 = field[iy,   ix+1]
    f01 = field[iy+1, ix  ]
    f11 = field[iy+1, ix+1]
    return (f00*(1-fx)*(1-fy) + f10*fx*(1-fy) + f01*(1-fx)*fy + f11*fx*fy)

# Example: differentiate w.r.t. lid speed
def loss_wrt_U(U):
    u0 = jnp.zeros((ny, nx))
    v0 = jnp.zeros((ny, nx))
    p0 = jnp.zeros((ny, nx))
    params = (dx, dy, dt, rho, nu, U)
    uT, vT, pT = simulate(u0, v0, p0, params)
    # simple scalar loss: mean kinetic energy at final time
    KE = 0.5*jnp.mean(uT**2 + vT**2)
    return KE

grad_U = jax.grad(loss_wrt_U)(U_lid)

print(f"Gradient of mean kinetic energy w.r.t. lid speed U: {grad_U}")