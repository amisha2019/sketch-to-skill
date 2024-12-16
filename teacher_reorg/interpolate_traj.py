import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_func.dataloader import get_dataloader
from data_func.dataloader_helper import rescale
from model.vae_mlp import VAE_MLP
import torch
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    # Load the checkpoint, which contains the model weights directly
    checkpoint = torch.load(args.model_path)
    
    # Create the model with the same architecture
    model = VAE_MLP(
        img_size=args.img_size, 
        in_channels=3, 
        latent_dim=args.latent_dim, 
        num_control_points=args.num_control_points, 
        degree=args.degree
    ).cuda()
    
    # Load the model's weights directly from the checkpoint
    model.load_state_dict(checkpoint)
    print("Loaded model weights from the checkpoint.")

    # Load the dataset (you only need the test set for interpolation)
    _, _, test_loader = get_dataloader(batch_size=args.bs, num_samples=None, img_size=args.img_size)

    # Interpolate between latent spaces and visualize the trajectories
    interpolate_latent_space(model, test_loader, args.output_dir, num_interpolations=args.num_interpolations)


def interpolate_latent_space(model, test_loader, output_dir, num_interpolations=10):
    model.eval()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    batch = next(iter(test_loader))
    sketch1, sketch2, traj_gt, params_gt, fitted_traj = batch
    sketch1, sketch2, traj_gt, params_gt, fitted_traj = sketch1.cuda(), sketch2.cuda(), traj_gt.cuda(), params_gt.cuda(), fitted_traj.cuda()
    
    # Encode the two trajectories into latent vectors
    with torch.no_grad():
        mu1, _ = model.encoder(sketch1)  # Encode trajectory 1
        mu2, _ = model.encoder(sketch2)  # Encode trajectory 2
        print(f"Shape of mu1: {mu1.shape}, Shape of mu2: {mu2.shape}")

    # Randomly pick two different indices
    batch_size = sketch1.size(0)
    idx1, idx2 = torch.randperm(batch_size)[:2]
    while idx1 == idx2:
        idx2 = torch.randint(0, batch_size, (1,))

    # Interpolate between the two latent vectors
    interpolated_latents_1 = []
    interpolated_latents_2 = []
    for alpha in np.linspace(0, 1, num_interpolations):
        print(f"alpha: {alpha}")
        interpolated_latents_1.append((1 - alpha) * mu1[idx1] + alpha * mu1[idx2])
        interpolated_latents_2.append((1 - alpha) * mu2[idx1] + alpha * mu2[idx2])
    interpolated_latents_1 = torch.stack(interpolated_latents_1, dim=0).cuda()
    interpolated_latents_2 = torch.stack(interpolated_latents_2, dim=0).cuda()
    print(f"Shape of interpolated_latents before MLP: {interpolated_latents_1.shape}")

    # Decode the interpolated latent vectors into trajectories and sketches
    with torch.no_grad():
        decoded_params = model.mlp(torch.cat((interpolated_latents_1, interpolated_latents_2), dim=1))
        generated_trajs = model.generate_trajectory(decoded_params).cpu().numpy()
        decoded_sketch1 = model.decoder(interpolated_latents_1).permute(0, 2, 3, 1).cpu().numpy()
        decoded_sketch2 = model.decoder(interpolated_latents_2).permute(0, 2, 3, 1).cpu().numpy()

    sketch1 = sketch1.permute(0, 2, 3, 1).cpu().numpy()
    sketch2 = sketch2.permute(0, 2, 3, 1).cpu().numpy()
    
    traj_gt_start = traj_gt[idx1].cpu().numpy()
    traj_gt_end = traj_gt[idx2].cpu().numpy()
    sketch_gt_1 = [sketch1[idx1], sketch1[idx2]]
    sketch_gt_2 = [sketch2[idx1], sketch2[idx2]]

    # Save the interpolated trajectories
    visualize_and_save_interpolated_trajectories(traj_gt_start, traj_gt_end, generated_trajs, sketch_gt_1, sketch_gt_2, decoded_sketch1, decoded_sketch2, output_dir)


def visualize_and_save_interpolated_trajectories(original_traj1, original_traj2, generated_trajs, sketch_gt_1, sketch_gt_2, decoded_sketch1, decoded_sketch2, output_dir):
    """
    Save the interpolated 3D trajectories
    """
    rows = 3
    num_interpolations = generated_trajs.shape[0]
    cols = num_interpolations + 2
    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    cmap = plt.get_cmap('viridis')

    # Plot original sketches
    ax = fig.add_subplot(rows, cols, 1)
    ax.imshow(rescale(sketch_gt_1[0]))
    ax.set_title("Original Sketch 1")
    ax.axis('off')

    ax = fig.add_subplot(rows, cols, cols + 1)
    ax.imshow(rescale(sketch_gt_2[0]))
    ax.set_title("Original Sketch 2")
    ax.axis('off')

    # Plot original trajectories
    ax = fig.add_subplot(rows, cols, 2*cols + 1, projection='3d')
    ax.scatter(original_traj1[:, 0], original_traj1[:, 1], original_traj1[:, 2], c=np.arange(len(original_traj1)), cmap=cmap, s=20)
    ax.plot(original_traj1[:, 0], original_traj1[:, 1], original_traj1[:, 2], label="Original Trajectory 1")
    ax.set_title("Original Trajectories 1")
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot interpolated trajectories
    for i in range(num_interpolations):
        # Plot decoded sketches
        ax = fig.add_subplot(rows, cols, i+2)
        ax.imshow(rescale(decoded_sketch1[i]))
        ax.set_title(f"Decoded Sketch 1 - {i+1}")
        ax.axis('off')

        ax = fig.add_subplot(rows, cols, cols+i+2)
        ax.imshow(rescale(decoded_sketch2[i]))
        ax.set_title(f"Decoded Sketch 2 - {i+1}")
        ax.axis('off')

        ax = fig.add_subplot(rows, cols, 2*cols+i+2, projection='3d')
        traj = generated_trajs[i]
        ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=np.arange(len(traj)), cmap=cmap, s=20)
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f"Interp {i+1}")
        ax.set_title(f"Interpolated Trajectory {i+1}")
        ax.legend()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    # Show sketch_gt_1 at the end of the first row
    ax = fig.add_subplot(rows, cols, cols)
    ax.imshow(rescale(sketch_gt_1[1]))
    ax.set_title("Original Sketch 1")
    ax.axis('off')

    # Show sketch_gt_2 at the end of the second row
    ax = fig.add_subplot(rows, cols, 2*cols)
    ax.imshow(rescale(sketch_gt_2[1]))
    ax.set_title("Original Sketch 2")
    ax.axis('off')

    # Plot original trajectories
    ax = fig.add_subplot(rows, cols, 3*cols, projection='3d')
    ax.scatter(original_traj2[:, 0], original_traj2[:, 1], original_traj2[:, 2], c=np.arange(len(original_traj2)), cmap=cmap, s=20)
    ax.plot(original_traj2[:, 0], original_traj2[:, 1], original_traj2[:, 2], label="Original Trajectory 2")
    ax.set_title("Original Trajectories 2")
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    output_file = os.path.join(output_dir, "interpolated_trajectories.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')  # Increase DPI and use tight bounding box
    plt.close()

    print(f"Saved interpolated trajectories to {output_file}")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="VAE MLP Latent Space Interpolation Script")
    
    # Provide the model architecture and other dynamic parameters
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input images')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space')
    parser.add_argument('--num_control_points', type=int, default=20, help='Number of control points')
    parser.add_argument('--degree', type=int, default=3, help='Degree of the B-spline')
    
    parser.add_argument('--model_path', type=str, default='', help='Path to the pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='', help='Directory to save the interpolated trajectories')
    parser.add_argument('--num_interpolations', type=int, default=10, help='Number of interpolation points between the two latent vectors')
    parser.add_argument('--bs', type=int, default=256, help='Batch size for loading data')

    args = parser.parse_args()

    args.model_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP/vae_mlp_2024-09-22_23-48-59_ep200_onecycle_lr0.001_bs256_kld0.0001_aug/models/vae_mlp_model_final.pth'
    args.output_dir = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP/vae_mlp_2024-09-22_23-48-59_ep200_onecycle_lr0.001_bs256_kld0.0001_aug/interpolate_traj'

    main(args)
