import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from scipy import interpolate
import numpy as np
from data_generator_v2 import generate_dataset

class SketchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256)
        )
    
    def forward(self, x):
        return self.cnn(x)

def subnet(ch_in, ch_out):
    return nn.Sequential(nn.Linear(ch_in, 512),
                         nn.ReLU(),
                         nn.Linear(512, ch_out))

class BSplineTrajectory2DCINN(nn.Module):
    def __init__(self, condition_dim=256, num_control_points=10, degree=3):
        super().__init__()
        self.encoder = SketchEncoder()
        self.cinn = self.build_inn(condition_dim, num_control_points)
        self.num_control_points = num_control_points
        self.degree = degree
        self.knots = self.create_uniform_knot_vector()
        
    def create_uniform_knot_vector(self):
        n_knots = self.num_control_points + self.degree + 1
        return torch.linspace(0, 1, n_knots)
    
    def build_inn(self, condition_dim, num_control_points):
        cond = Ff.ConditionNode(condition_dim)
        nodes = [Ff.InputNode(num_control_points, 2)]

        for k in range(8):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': 1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)
    
    def forward(self, sketch, control_points=None):
        condition = self.encoder(sketch)
        
        if self.training:
            z, log_jac_det = self.cinn(control_points, c=condition)
            return z, log_jac_det
        else:
            z = torch.randn(sketch.shape[0], self.num_control_points, 2).to(sketch.device)
            control_points = self.cinn(z, c=condition, rev=True)
            return control_points
    
    def generate_trajectory(self, control_points, num_points=100):
        # Convert to numpy for scipy
        control_points_np = control_points.detach().cpu().numpy()
        knots_np = self.knots.detach().cpu().numpy()
        
        tck = (knots_np, [control_points_np[0, :, 0], control_points_np[0, :, 1]], self.degree)
        t = np.linspace(knots_np[self.degree], knots_np[-self.degree-1], num_points)
        trajectory_np = interpolate.splev(t, tck)
        
        # Convert back to torch
        return torch.tensor(np.array(trajectory_np).T, dtype=torch.float32).to(control_points.device)

def fit_bspline_trajectory(points, num_control_points=10, degree=3):
    t = np.linspace(0, 1, len(points))
    tck, _ = interpolate.splprep([points[:, 0], points[:, 1]], s=0, k=degree)
    knots, coeffs, degree = tck
    
    # Resample to get the desired number of control points
    new_knots = np.linspace(knots.min(), knots.max(), num_control_points + degree + 1)
    control_points = interpolate.splev(new_knots[degree:-degree], tck)
    
    return torch.tensor(np.array(control_points).T, dtype=torch.float32)

def train(model, optimizer, dataloader, num_epochs):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            sketch, traj = batch
            
            # Fit B-spline to the trajectory
            control_points = fit_bspline_trajectory(traj, num_control_points=model.num_control_points, degree=model.degree)
            
            optimizer.zero_grad()
            z, log_jac_det = model(sketch, control_points)
            
            # Calculate negative log-likelihood loss
            loss = -torch.mean(log_jac_det)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Main execution
if __name__ == "__main__":
    # Generate dataset
    sketches, trajectories = generate_dataset(num_samples=1000, max_displacement=2.0)

    # Convert to PyTorch tensors
    sketches_tensor = torch.from_numpy(sketches).float().unsqueeze(1)  # Add channel dimension
    trajectories_tensor = torch.from_numpy(trajectories).float()

    # Create data loader
    dataset = torch.utils.data.TensorDataset(sketches_tensor, trajectories_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Initialize and train the model
    model = BSplineTrajectory2DCINN(num_control_points=10, degree=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train(model, optimizer, dataloader, num_epochs=100)

    # Inference example
    sketch = torch.randn(1, 1, 224, 224)  # Assumed input size
    control_points = model(sketch)
    generated_trajectory = model.generate_trajectory(control_points)

    # Visualize the generated trajectory
    import matplotlib.pyplot as plt

    def visualize_trajectory(trajectory, control_points):
        plt.figure(figsize=(10, 10))
        plt.plot(trajectory[0, :, 0].detach().cpu().numpy(), 
                 trajectory[0, :, 1].detach().cpu().numpy(), label='Generated Trajectory')
        plt.scatter(control_points[0, :, 0].detach().cpu().numpy(), 
                    control_points[0, :, 1].detach().cpu().numpy(), color='red', label='Control Points')
        plt.title("Generated 2D Trajectory with B-spline")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()

    visualize_trajectory(generated_trajectory, control_points)