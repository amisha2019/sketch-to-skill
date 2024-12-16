import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import VAE_CINN
from data import get_dataloader

def visualize_trajectory(sketches, recons, trajectories, fitted_trajectories, generated_trajectories, img_name):
    num_samples = len(sketches)
    fig, axes = plt.subplots(5, num_samples, figsize=(3*num_samples, 6))
    
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

        axes[4, i].imshow(np.flipud(recons[i].squeeze()), cmap='gray')
        axes[4, i].set_title(f"Sample {i+1}: Reconstructed Sketch")
        axes[4, i].set_aspect('equal', 'box')
    
    plt.tight_layout()
    plt.savefig(f'/fs/nexus-scratch/amishab/Teacher_student_RLsketch/eval_results2/generated_trajectory_{img_name}.png')

def evaluate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        i = 0
        for sketch, traj, params_gt, fitted_traj in dataloader:
            sketches = []
            reconstructed_sketches = []
            trajectories = []
            generated_trajectories = []
            fitted_trajectories = []
            for _ in range(6):
                sketch = sketch.cuda()
                params, recons = model(sketch)

                generated_trajectory = model.generate_trajectory(params)
                generated_trajectories.append(generated_trajectory.cpu().numpy().squeeze())

                sketches.append(sketch.cpu().numpy().squeeze())
                reconstructed_sketches.append(recons.cpu().numpy().squeeze())
                trajectories.append(traj.numpy().squeeze())
                fitted_trajectories.append(fitted_traj.numpy().squeeze())
            
            visualize_trajectory(sketches, reconstructed_sketches, trajectories, fitted_trajectories, generated_trajectories, img_name=i)
            i += 1

if __name__ == "__main__":
    # root_dir = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/{os.environ.get('USER')}"

    img_size = 64
    model = VAE_CINN(img_size=img_size,
                    in_channels=1,
                    latent_dim=128,
                    condition_dim=64,
                    num_control_points=10,
                    degree=3,).cuda()
    # model.load_state_dict(torch.load(f'{root_dir}/bspline_cinn_model_64_2024-09-12_19-33-43.pth'))
    model.load_state_dict(torch.load('/fs/nexus-projects/Sketch_VLM_RL/teacher_model/asingh/2024-09-12_23-25-45_bs128_ns200000_lr0.0001_kld0.01_epochs500/bspline_cinn_model_bs128_ns200000_lr0.0001_kld0.01_epochs500.pth'))
    
    dataloader = get_dataloader(batch_size=1, num_samples=6, img_size=img_size)  # Small batch for visualization
    evaluate_model(model, dataloader)
