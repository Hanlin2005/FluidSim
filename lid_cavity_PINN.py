import torch
import numpy as np
from torch import nn
from lid_cavity_FDM import simulate, interpolate_solution
import matplotlib.pyplot as plt
import matplotlib.cm as cm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device:", device)

#Hyperparamters
num_epochs = 5000
learning_rate = 0.001
batch_size = 64

#Fluid problsm parameters
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


class LidPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.structure = nn.Sequential(
            nn.Linear(2, 20),  # Input layer with 2 features (x, y)
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 20),
            nn.Tanh(),
            nn.Linear(20, 3)   # Output layer with 3 features (u, v, p)
        )
    
    def forward(self, x):
        return self.structure(x)
    
def loss_function(model, interior_points,):
    #get loss on interior points
    interior_points.requires_grad_(True)
    interior_x = interior_points[:, 0:1]
    interior_y = interior_points[:, 1:2]
    interior_points.requires_grad_(True)

    #get model predictions
    predictions = model(interior_points)
    u, v, p = predictions[:, 0:1], predictions[:, 1:2], predictions[:, 2:3]

    #Find grads
    u_grads = torch.autograd.grad(u, interior_points, torch.ones_like(u), create_graph=True)[0]
    u_x, u_y = u_grads[:, 0:1], u_grads[:, 1:2]
    v_grads = torch.autograd.grad(v, interior_points, torch.ones_like(v), create_graph=True)[0]
    v_x, v_y = v_grads[:, 0:1], v_grads[:, 1:2]
    p_grads = torch.autograd.grad(p, interior_points, torch.ones_like(p), create_graph=True)[0]
    p_x, p_y = p_grads[:, 0:1], p_grads[:, 1:2]

    # Second derivatives
    u_xx = torch.autograd.grad(u_x, interior_points, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(u_y, interior_points, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]
    v_xx = torch.autograd.grad(v_x, interior_points, torch.ones_like(v_x), create_graph=True)[0][:, 0:1]
    v_yy = torch.autograd.grad(v_y, interior_points, torch.ones_like(v_y), create_graph=True)[0][:, 1:2]

    # Compute residuals of the Navier-Stokes equations
    residual_x_momentum = (u * u_x + v * u_y) + (1/rho) * p_x - nu * (u_xx + u_yy)
    residual_y_momentum = (u * v_x + v * v_y) + (1/rho) * p_y - nu * (v_xx + v_yy)
    residual_continuity = u_x + v_y

    # Compute the loss as the mean squared error of the residuals
    loss_interior = nn.MSELoss()(residual_x_momentum, torch.zeros_like(residual_x_momentum)) + \
                    nn.MSELoss()(residual_y_momentum, torch.zeros_like(residual_y_momentum)) + \
                    nn.MSELoss()(residual_continuity, torch.zeros_like(residual_continuity))
    return loss_interior

#return random sample points, normalized to length of 1
def sample_points(n_interior, n_boundary):
    interior_points = torch.rand(n_interior, 2)
    boundary_points = {
        'top': torch.cat([torch.rand(n_boundary, 1), torch.ones(n_boundary, 1)], dim=1),
        'bottom': torch.cat([torch.rand(n_boundary, 1), torch.zeros(n_boundary, 1)], dim=1),
        'left': torch.cat([torch.zeros(n_boundary, 1), torch.rand(n_boundary, 1)], dim=1),
        'right': torch.cat([torch.ones(n_boundary, 1), torch.rand(n_boundary, 1)], dim=1)
    }
    return interior_points, boundary_points

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_pinn_solution(model, X, Y, u, v, p, device):
    """
    Takes a trained PINN and visualizes its predictions for u, v, and p.

    Args:
        model (nn.Module): The trained PyTorch model.
        X (np.array): The meshgrid for the x-coordinates.
        Y (np.array): The meshgrid for the y-coordinates.
        device (torch.device): The device the model is on.
    """
    # 1. Set the model to evaluation mode
    model.eval()

    # 2. Prepare the grid of points to pass to the model
    # Flatten the grid and create a tensor of (x, y) coordinates
    x_flat = torch.tensor(X.flatten(), dtype=torch.float32)
    y_flat = torch.tensor(Y.flatten(), dtype=torch.float32)
    grid_points = torch.stack([x_flat, y_flat], dim=1).to(device)

    # 3. Get predictions from the model
    with torch.no_grad(): # Disable gradient calculation for efficiency
        predictions = model(grid_points)

    # 4. Process the predictions
    # Move predictions to CPU, convert to NumPy, and separate u, v, p
    pred_np = predictions.cpu().numpy()
    u_pred = pred_np[:, 0].reshape(X.shape)
    v_pred = pred_np[:, 1].reshape(X.shape)
    p_pred = pred_np[:, 2].reshape(X.shape)

    # 5. Create the plot (using the same format as your FDM script)
    plt.figure(figsize=(12, 5))

    # Plot Pressure contour
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, p, alpha=0.8, cmap=cm.viridis, levels=50)
    plt.streamplot(X, Y, u, v, color='black', density=1.5)
    plt.colorbar(label='Pressure (p)')
    plt.title('Simulated Velocity and Pressure Fields')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Plot Velocity streamlines
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, p_pred, alpha=0.8, cmap=cm.viridis, levels=50)
    plt.streamplot(X, Y, u_pred, v_pred, color='black', density=1.5)
    plt.colorbar(label='Pressure (p)')
    plt.title('PINN Predicted Velocity and Pressure Fields')
    plt.xlabel('X-axis')

    plt.tight_layout()
    plt.show()

def evaluate_error(model, X, Y, x, y, u,v,p):
    model.eval()
    grid_points = torch.stack([torch.tensor(X.flatten(), dtype=torch.float32), torch.tensor(Y.flatten(), dtype=torch.float32)], dim=1).to(device)

    with torch.no_grad():
        predictions = model(grid_points).cpu()
    
    true_values = torch.from_numpy(interpolate_solution(grid_points.cpu(), u, v, p, x, y))

    error = (torch.norm(predictions - true_values) / (torch.norm(true_values)+1e-12)).item()  #.item to make it into a float from a 1 element tensor

    pointwise_rel = torch.abs(predictions-true_values) / (torch.abs(true_values) + 1e-12) #pointwise relative error

    return error, pointwise_rel.reshape(X.shape[0], X.shape[1], 3).cpu().numpy()


def plot_error_map(X, Y, pointwise_rel, titles=("u", "v", "p")):
    """
    pointwise_rel: (ny, nx, 3) relative error field from evaluate_error
    """

    figcols = 4
    plt.figure(figsize=(4*figcols, 4))

    # Per-field heatmaps
    for i in range(3):
        plt.subplot(1, figcols, i+1)
        plt.contourf(X, Y, pointwise_rel[..., i], levels=50, cmap=cm.inferno)
        plt.colorbar(label="relative error")
        plt.title(f"Rel. Error: {titles[i]}")
        plt.xlabel("x"); plt.ylabel("y")

    # Combined error (L2 across fields)
    comb = np.linalg.norm(pointwise_rel, axis=-1)  # sqrt(u^2+v^2+p^2) pointwise (relative)
    plt.subplot(1, figcols, 4)
    plt.contourf(X, Y, comb, levels=50, cmap=cm.inferno)
    plt.colorbar(label="relative error (L2 over fields)")
    plt.title("Combined Rel. Error (u,v,p)")
    plt.xlabel("x"); plt.ylabel("y")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    #Simulate fluid flow
    u, v, p = simulate(nt, u, v, dt, dx, dy, p, rho, nu, nx, ny, nit)

    #Training
    model = LidPINN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train()


    print("Starting training...")
    for epoch in range(num_epochs):
        interior_points, boundary_points = sample_points(1000, 100)
        scale = torch.tensor((2.0, 2.0), dtype=torch.float32, device=device)
        interior_points = interior_points.to(device) * scale
        boundary_points = torch.cat([val.to(device) for val in boundary_points.values()], dim=0) * scale

        optimizer.zero_grad()
        pred_interior = model(interior_points)
        pred_boundary = model(boundary_points)

        #Find true values for interior and boundary points
        true_interior = torch.tensor(interpolate_solution(interior_points.cpu().numpy(), u, v, p, x, y), dtype=torch.float32).to(device)
        true_boundary = torch.tensor(interpolate_solution(boundary_points.cpu().numpy(), u, v, p, x, y), dtype=torch.float32).to(device)

        #Experimenting with different loss functions
        loss = loss_function(model, interior_points) + nn.MSELoss()(pred_boundary, true_boundary)
        #loss = nn.MSELoss()(pred_interior, true_interior) + nn.MSELoss()(pred_boundary, true_boundary)

        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    # ... (end of your training loop) ...
    print("Training finished.")

    torch.save(model, "lid_pinn_full.pth")

    # Call the visualization function with the trained model
    visualize_pinn_solution(model, X, Y, u, v, p, device)