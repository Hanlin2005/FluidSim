#This code simulates the lid cavity problem using a finite difference method
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#Initialize variables
grid_size = (2,2)
nx = 41         #number of x cells
ny = 41         #number of y cells
nt = 500        #number of time steps
nit = 50
dx = grid_size[0] / (nx-1)  #find dx
dy = grid_size[1] / (ny-1)  #find dy
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)

rho = 1 #density
nu = 0.1
dt = 0.001

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx)) 
b = np.zeros((ny, nx))

def build_up_b(b, rho, dt, u, v, dx, dy):
    
    b[1:-1, 1:-1] = (rho * (1 / dt * 
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
                     (2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                    ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
                      2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                           (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
                          ((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

    return b

def pressure_poisson(p, dx, dy, b):
    pn = np.empty_like(p)
    pn = p.copy()
    
    for q in range(nit):
        pn = p.copy()
        p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
                          (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
                          (2 * (dx**2 + dy**2)) -
                          dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
                          b[1:-1,1:-1])

        p[:, -1] = p[:, -2] # dp/dx = 0 at x = 2
        p[0, :] = p[1, :]   # dp/dy = 0 at y = 0
        p[:, 0] = p[:, 1]   # dp/dx = 0 at x = 0
        p[-1, :] = 0        # p = 0 at y = 2
        
    return p

def simulate(nt, u, v, dt, dx, dy, pn, rho, nu):
    b = np.zeros((ny, nx))

    for n in range(nt):
        print("Simulating step " + str(n))

        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        pn = pressure_poisson(pn, dx, dy, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[1:-1, :-2]) 
                        - vn[1:-1, 1:-1] * dt/dx * (un[1:-1, 1:-1] - un[:-2, 1:-1]) - dt/(2 * rho * dx) * (pn[1:-1, 2:] - pn[1:-1, :-2])
                        + nu * (dt/(dx**2) * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, :-2]) + 
                        dt/(dy**2) * (un[2:,1:-1] - 2 * un[1:-1, 1:-1] + un[:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
                        un[1:-1, 1:-1] * dt / dx *
                        (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
                        vn[1:-1, 1:-1] * dt / dy *
                        (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
                        dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                        nu * (dt / dx**2 *
                        (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                        dt / dy**2 *
                        (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))
        

        u[0,:] = 0
        u[-1,:] = 1
        u[:,0] = 0
        u[:,-1] = 0
        v[0,:] = 0
        v[-1,:] = 0
        v[:,0] = 0
        v[:,-1] = 0
    
    return u,v,p


u, v, p = simulate(nt, u, v, dt, dx, dy, p, rho, nu)


plt.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)

# Add a color bar to show what pressure values the colors correspond to
plt.colorbar(label='Pressure')

# --- 3. Plot the velocity field ---
# Draw streamlines to show the direction of the flow (u, v)
# The density parameter controls how many lines are drawn.
plt.streamplot(X, Y, u, v, color='black', density=1.2)
# An alternative is a quiver plot, which draws arrows:
# plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])

# --- 4. Final touches ---
plt.title('Lid-Driven Cavity Flow')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Display the plot
plt.show()