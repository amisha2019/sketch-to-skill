import torch
import numpy as np
import matplotlib.pyplot as plt
from model import BSplineTrajectory2DCINN
# from data import get_dataloader
from data_real import get_dataloader

def visualize_trajectory(sketches, fitted_trajectories, generated_trajectories, img_name):
    num_samples = len(sketches)
    fig, axes = plt.subplots(3, num_samples, figsize=(3*num_samples, 6))
    
    for i in range(num_samples):
        sketch = (sketches[i].squeeze().transpose(1, 2, 0) * 255).astype(int)
        # breakpoint()
        axes[0, i].imshow(sketch)
        axes[0, i].set_title(f"Sample {i+1}: Sketch")
        axes[0, i].set_aspect('equal', 'box')

        axes[1, i].set_aspect('equal', 'box')
        axes[1, i].plot(fitted_trajectories[i][:, 0], fitted_trajectories[i][:, 1])
        axes[1, i].set_title(f"Sample {i+1}: Fitted Trajectory")
        axes[1, i].set_aspect('equal', 'box')
        # axes[1, i].set_xlim(-1, 1)
        # axes[1, i].set_ylim(-1, 1)

        axes[2, i].set_aspect('equal', 'box')
        axes[2, i].plot(generated_trajectories[i][:, 0], generated_trajectories[i][:, 1])
        axes[2, i].set_title(f"Sample {i+1}: Generated Trajectory")
        axes[2, i].set_aspect('equal', 'box')
        # axes[2, i].set_xlim(-1, 1)
        # axes[2, i].set_ylim(-1, 1)
    
    plt.tight_layout()
    plt.savefig(f'generated_trajectory_real_{img_name}.png')

def evaluate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        i = 0
        for sketch, params, fitted_traj in dataloader:
            sketches = []
            generated_trajectories = []
            fitted_trajectories = []
            for _ in range(6):
                sketch = sketch.cuda()
                # breakpoint()
                params = model(sketch)
                params = params[0]

                generated_trajectory = model.generate_trajectory(params).squeeze()
                generated_trajectories.append(generated_trajectory)
                # breakpoint()

                sketches.append(sketch.cpu().numpy().squeeze())
                # trajectories.append(traj.numpy().squeeze())
                fitted_trajectories.append(fitted_traj.numpy().squeeze())
                
            visualize_trajectory(sketches, fitted_trajectories, generated_trajectories, img_name=i)
            i += 1

if __name__ == "__main__":
    root_dir = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong"

    model = BSplineTrajectory2DCINN(num_control_points=10, degree=3).cuda()
    model.load_state_dict(torch.load(f'{root_dir}/bspline_cinn_model_2D_CINN_real.pth'))
    
    dataloader = get_dataloader(batch_size=1, num_samples=6)  # Small batch for visualization
    evaluate_model(model, dataloader)