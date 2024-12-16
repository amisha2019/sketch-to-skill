import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import VAE_CINN
from data import get_dataloader

def unnormalize_sketch(sketch):
    data_mean = 0.0037
    data_std = 0.0472
    data_max = 22.0
    return sketch * data_max * data_std + data_mean

def visualize_trajectory(sketches1, sketches2, reconstructed_sketches1, reconstructed_sketches2, trajectories, fitted_trajectories, generated_trajectories, img_name):
    num_samples = len(sketches1)
    fig, axes = plt.subplots(figsize=(6*num_samples, 36))

    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    for i in range(num_samples):
        ax1 = fig.add_subplot(6, num_samples, i+1)
        ax1.imshow(np.flipud(unnormalize_sketch(sketches1[i].squeeze())))
        ax1.set_title(f"Sample {i+1}: Sketch 1")
        ax1.set_aspect('equal', 'box')

        ax2 = fig.add_subplot(6, num_samples, num_samples+i+1)
        ax2.imshow(np.flipud(unnormalize_sketch(sketches2[i].squeeze())))
        ax2.set_title(f"Sample {i+1}: Sketch 2")
        ax2.set_aspect('equal', 'box')

        ax3 = fig.add_subplot(6, num_samples, 2*num_samples+i+1)
        ax3.imshow(np.flipud(unnormalize_sketch(reconstructed_sketches1[i].squeeze())))
        ax3.set_title(f"Sample {i+1}: Reconstructed Sketch 1")
        ax3.set_aspect('equal', 'box')

        ax4 = fig.add_subplot(6, num_samples, 3*num_samples+i+1)
        ax4.imshow(np.flipud(unnormalize_sketch(reconstructed_sketches2[i].squeeze())))
        ax4.set_title(f"Sample {i+1}: Reconstructed Sketch 2")
        ax4.set_aspect('equal', 'box')

        ax5 = fig.add_subplot(6, num_samples, 4*num_samples+i+1, projection='3d')
        ax5.scatter(trajectories[i][:, 0], trajectories[i][:, 1], trajectories[i][:, 2], c=np.arange(len(trajectories[i])), cmap=cmap, alpha=0.3)
        ax5.plot(fitted_trajectories[i][:, 0], fitted_trajectories[i][:, 1], fitted_trajectories[i][:, 2], 'r-', linewidth=2)
        ax5.set_xlabel('X')
        ax5.set_ylabel('Y')
        ax5.set_zlabel('Z')
        ax5.set_title(f"Sample {i+1}: Fitted Trajectory")

        ax6 = fig.add_subplot(6, num_samples, 5*num_samples+i+1, projection='3d')
        ax6.plot(generated_trajectories[i][:, 0], generated_trajectories[i][:, 1], generated_trajectories[i][:, 2])
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        ax6.set_title(f"Sample {i+1}: Generated Trajectory")
        
    plt.tight_layout()
    plt.savefig(f'eval_results/vae_cinn_2024-09-15_16-54-47_generated_trajectory_{img_name}.png')

def evaluate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        i = 0
        for sketch1, sketch2, traj, params_gt, fitted_traj in dataloader:
            sketches1 = []
            sketches2 = []
            reconstructed_sketches1 = []
            reconstructed_sketches2 = []
            trajectories = []
            generated_trajectories = []
            fitted_trajectories = []
            for _ in range(6):
                sketch1 = sketch1.cuda()
                sketch2 = sketch2.cuda()
                params, recons1, recons2 = model(sketch1, sketch2)

                generated_trajectory = model.generate_trajectory(params)
                generated_trajectories.append(generated_trajectory.cpu().numpy().squeeze())

                sketches1.append(sketch1.cpu().numpy().squeeze().transpose(1, 2, 0))
                sketches2.append(sketch2.cpu().numpy().squeeze().transpose(1, 2, 0))
                reconstructed_sketches1.append(recons1.cpu().numpy().squeeze().transpose(1, 2, 0))
                reconstructed_sketches2.append(recons2.cpu().numpy().squeeze().transpose(1, 2, 0))

                trajectories.append(traj.numpy().squeeze())
                fitted_trajectories.append(fitted_traj.numpy().squeeze())
            
            visualize_trajectory(sketches1, sketches2, reconstructed_sketches1, reconstructed_sketches2, trajectories, fitted_trajectories, generated_trajectories, img_name=i)
            i += 1

if __name__ == "__main__":
    model_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_3D/vae_cinn_2024-09-15_16-54-47/bspline_cinn_model_64.pth"

    img_size = 64
    num_control_points = 20
    model = VAE_CINN(img_size=img_size,
                    in_channels=3,
                    latent_dim=128,
                    condition_dim=64,
                    num_control_points=num_control_points,
                    degree=3,).cuda()
    model.load_state_dict(torch.load(model_path))
    
    dataloader, _, _ = get_dataloader(batch_size=1, num_samples=6, img_size=img_size, val_split=0, test_split=0)  # Small batch for visualization
    evaluate_model(model, dataloader)
