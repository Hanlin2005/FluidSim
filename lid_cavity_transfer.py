#THIS CODE IS FOR THE TRANSFER LEARNING EXPERIMENTS OF THE LID CAVITY BENCHMARK
import torch
import numpy as np
from torch import nn
from lid_cavity_FDM import simulate, interpolate_solution, display_solution
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from lid_cavity_PINN import LidPINN, loss_function, visualize_pinn_solution, sample_points

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#VARIABLES for new problem
grid_size = (2,2)
nx = 41       #number of x cells
ny = 41         #number of y cells
nt = 500        #number of time steps
nit = 50
dx = grid_size[0] / (nx-1)  #find dx
dy = grid_size[1] / (ny-1)  #find dy
x = np.linspace(0, 2, nx)
y = np.linspace(0, 2, ny)
X, Y = np.meshgrid(x, y)
U = 10 #lid velocity

rho = 1 #density
nu = 0.1
dt = 0.001

u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx)) 
b = np.zeros((ny, nx))

lr = 0.0001
epochs = 500
n_interior = 1024
n_boundary = 256

#Load the pre-trained model
model = torch.load("lid_pinn_full.pth", map_location=device)
model.eval()


#Simulate the new problem and display the solution
u, v, p = simulate(nt, u, v, dt, dx, dy, p, rho, nu, nx, ny, nit, U)
display_solution(X, Y, u, v, p)

#Fine-tune the model
model.train()

freeze_until = 0  # freeze the first N Linear layers; tune as you like (0 = freeze none)
lin_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
for i, layer in enumerate(lin_layers):
    if i < freeze_until:
        for p_ in layer.parameters():
            p_.requires_grad = False

optimizer = torch.optim.Adam(filter(lambda p_: p_.requires_grad, model.parameters()), lr=lr)

model.train()
for ep in range(epochs):
    interior_pts, boundary_pts = sample_points(n_interior, n_boundary)

    scale = torch.tensor((2.0, 2.0), dtype=torch.float32, device=device)
    interior_pts = interior_pts.to(device) * scale
    boundary_pts = torch.cat([val.to(device) for val in boundary_pts.values()], dim=0) * scale
    # targets from FDM
    with torch.no_grad():
        true_int_np = interpolate_solution(interior_pts.detach().cpu().numpy(), u, v, p, x, y)
        true_bnd_np = interpolate_solution(boundary_pts.detach().cpu().numpy(), u, v, p, x, y)
    true_int = torch.tensor(true_int_np, dtype=torch.float32, device=device)
    true_bnd = torch.tensor(true_bnd_np, dtype=torch.float32, device=device)

    # predictions
    pred_bnd = model(boundary_pts)
    # interior PDE residual
    res_loss = loss_function(model, interior_pts)

    # small supervised interior data (optional; helps stabilize/accelerate adaptation)
    pred_int = model(interior_pts)
    data_int_loss = nn.MSELoss()(pred_int, true_int)

    # boundary data term
    bnd_loss = nn.MSELoss()(pred_bnd, true_bnd)

    # total loss
    w_res = 1.0
    w_bnd = 1.0
    w_data_int = 0.1

    #loss = w_res*res_loss + w_bnd*bnd_loss + w_data_int*data_int_loss
    loss = data_int_loss + bnd_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ep % 100 == 0:
        print(f"[U={U}] Epoch {ep:4d} | "
                f"Loss {loss.item():.4e}  (res {res_loss.item():.4e}, bnd {bnd_loss.item():.4e}, int {data_int_loss.item():.4e})")

#Evaluate the fine-tuned model
model.eval()
visualize_pinn_solution(model, X, Y, u, v, p, device)