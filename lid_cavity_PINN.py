import torch
import numpy as np
from torch import nn
from lid_cavity_FDM import simulate

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
            nn.Linear(2, 50),  # Input layer with 2 features (x, y)
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 3)   # Output layer with 3 features (u, v, p)
        )
    
    def forward(self, x):
        return self.structure(x)
    
def loss_function(pred, true):
    return nn.MSELoss()(pred, true)

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

def visualize_pinn_solution(model, X, Y, device):
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
    plt.contourf(X, Y, p_pred, alpha=0.8, cmap=cm.viridis, levels=50)
    plt.colorbar(label='Pressure (p)')
    plt.title('PINN Predicted Pressure')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # Plot Velocity streamlines
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, p_pred, alpha=0.8, cmap=cm.viridis, levels=50)
    plt.streamplot(X, Y, u_pred, v_pred, color='black', density=1.5)
    plt.title('PINN Predicted Velocity')
    plt.xlabel('X-axis')

    plt.tight_layout()
    plt.show()


#Simulate fluid flow
u, v, p = simulate(nt, u, v, dt, dx, dy, p, rho, nu)

#Training
model = LidPINN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()


print("Starting training...")
for epoch in range(num_epochs):
    interior_points, boundary_points = sample_points(1000, 100)
    interior_points = interior_points.to(device)
    boundary_points = torch.cat([val.to(device) for val in boundary_points.values()], dim=0)

    optimizer.zero_grad()
    pred_interior = model(interior_points)
    pred_boundary = model(boundary_points)

    #Find true values for interior and boundary points
    u_interior = [u[point[0],point[1]] for point in np.array(interior_points.cpu().numpy() * [nx-1,ny-1]).astype(int)]
    v_interior = [v[point[0],point[1]] for point in np.array(interior_points.cpu().numpy() * [nx-1,ny-1]).astype(int)]
    p_interior = [p[point[0],point[1]] for point in np.array(interior_points.cpu().numpy() * [nx-1,ny-1]).astype(int)]
    u_boundary = [u[point[0],point[1]] for point in np.array(boundary_points.cpu().numpy() * [nx-1,ny-1]).astype(int)]
    v_boundary = [v[point[0],point[1]] for point in np.array(boundary_points.cpu().numpy() * [nx-1,ny-1]).astype(int)]
    p_boundary = [p[point[0],point[1]] for point in np.array(boundary_points.cpu().numpy() * [nx-1,ny-1]).astype(int)]
    true_interior = torch.tensor(np.array([u_interior, v_interior, p_interior]).T, dtype=torch.float32).to(device)
    true_boundary = torch.tensor(np.array([u_boundary, v_boundary, p_boundary]).T, dtype=torch.float32).to(device)

    loss = loss_function(torch.cat([pred_interior, pred_boundary], dim=0),
                            torch.cat([true_interior, true_boundary], dim=0))

    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# ... (end of your training loop) ...
print("Training finished.")

# Call the visualization function with the trained model
visualize_pinn_solution(model, X, Y, device)