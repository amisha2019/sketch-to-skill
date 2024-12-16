import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SimplifiedBSplineModel
from data import get_dataloader
from scipy import interpolate

def generate_trajectory(params, num_control_points=10, degree=3, num_points=100):
    """ Generates a B-spline trajectory from control points """
    control_points = params.reshape(num_control_points, 2)
    
    # Create knot vector
    knots = np.linspace(0, 1, num_control_points + degree + 1)
    knots[:degree+1] = 0
    knots[-degree-1:] = 1
    
    # Create B-spline
    tck = [knots, [control_points[:, 0], control_points[:, 1]], degree]
    u = np.linspace(0, 1, num_points)
    spline = interpolate.splev(u, tck)
    
    return np.array(spline).T

def visualize_trajectory(sketches, trajectories, fitted_trajectories, generated_trajectories, img_name):
    num_samples = len(sketches)
    fig, axes = plt.subplots(4, num_samples, figsize=(3*num_samples, 6))
    
    for i in range(num_samples):
        axes[0, i].imshow(np.flipud(sketches[i].squeeze()), cmap='gray')
        axes[0, i].set_title(f"Sample {i+1}: Sketch")
        axes[1, i].set_aspect('equal', 'box')
        
        axes[1, i].plot(trajectories[i][:, 0], trajectories[i][:, 1])
        axes[1, i].set_title(f"Sample {i+1}: Trajectory")
        axes[1, i].set_aspect('equal', 'box')
        axes[1, i].set_xlim(-1, 1)
        axes[1, i].set_ylim(-1, 1)

        axes[2, i].set_aspect('equal', 'box')
        axes[2, i].plot(fitted_trajectories[i][:, 0], fitted_trajectories[i][:, 1])
        axes[2, i].set_title(f"Sample {i+1}: Fitted Trajectory")
        axes[2, i].set_aspect('equal', 'box')
        axes[2, i].set_xlim(-1, 1)
        axes[2, i].set_ylim(-1, 1)

        axes[3, i].set_aspect('equal', 'box')
        axes[3, i].plot(generated_trajectories[i][:, 0], generated_trajectories[i][:, 1])
        axes[3, i].set_title(f"Sample {i+1}: Generated Trajectory")
        axes[3, i].set_aspect('equal', 'box')
        axes[3, i].set_xlim(-1, 1)
        axes[3, i].set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(f'generated_trajectory_{img_name}.png')

def evaluate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        i = 0
        for sketch, traj, params, fitted_traj in dataloader:
            sketches = []
            trajectories = []
            generated_trajectories = []
            fitted_trajectories = []
            for _ in range(6):
                sketch = sketch.cuda()
                
                # Predict control points
                predicted_params = model(sketch)
                predicted_params = predicted_params[0].cpu().numpy()  # Move to CPU and convert to numpy
                
                # Generate trajectory using predicted control points
                generated_trajectory = generate_trajectory(predicted_params)
                generated_trajectories.append(generated_trajectory)
                
                # Store data for visualization
                sketches.append(sketch.cpu().numpy().squeeze())
                trajectories.append(traj.numpy().squeeze())
                fitted_trajectories.append(fitted_traj.numpy().squeeze())
                
            visualize_trajectory(sketches, trajectories, fitted_trajectories, generated_trajectories, img_name=i)
            i += 1

if __name__ == "__main__":
    root_dir = "/fs/nexus-scratch/amishab/Teacher_student_RLsketch/saved_models"

    model = SimplifiedBSplineModel(num_control_points=10).cuda()
    model.load_state_dict(torch.load(f'{root_dir}/bspline_mlp_model_2D.pth'))
    
    dataloader = get_dataloader(batch_size=1, num_samples=6)  # Small batch for visualization
    evaluate_model(model, dataloader)
