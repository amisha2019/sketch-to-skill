import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from CAE_INN import VAE_CINN, ConvEncoder, ConvDecoder  # Ensure you import ConvEncoder and ConvDecoder
from data import get_dataloader
from scipy import interpolate
import os

# Function to visualize results
def visualize_trajectory(sketches, trajectories, fitted_trajectories, generated_trajectories, img_name):
    num_samples = len(sketches)
    fig, axes = plt.subplots(4, num_samples, figsize=(3 * num_samples, 6))

    # Normalize generated trajectories for visualization
    generated_trajectories_normalized = []
    for trajectory in generated_trajectories:
        max_val = np.max(np.abs(trajectory))
        normalized_trajectory = trajectory / max_val  # Normalize to [-1, 1]
        generated_trajectories_normalized.append(normalized_trajectory)

    for i in range(num_samples):
        axes[0, i].imshow(np.flipud(sketches[i].reshape(64, 64)), cmap='gray')
        axes[0, i].set_title(f"Sample {i+1}: Sketch")

        axes[1, i].plot(trajectories[i][:, 0], trajectories[i][:, 1])
        axes[1, i].set_title(f"Sample {i+1}: Trajectory")
        axes[1, i].set_xlim(-1, 1)
        axes[1, i].set_ylim(-1, 1)
        axes[1, i].set_aspect('equal', 'box')

        axes[2, i].plot(fitted_trajectories[i][:, 0], fitted_trajectories[i][:, 1])
        axes[2, i].set_title(f"Sample {i+1}: Fitted Trajectory")
        axes[2, i].set_xlim(-1, 1)
        axes[2, i].set_ylim(-1, 1)
        axes[2, i].set_aspect('equal', 'box')

        # Plot normalized generated trajectory
        axes[3, i].plot(generated_trajectories_normalized[i][:, 0], generated_trajectories_normalized[i][:, 1])
        axes[3, i].set_title(f"Sample {i+1}: Generated Trajectory")
        axes[3, i].set_xlim(-1, 1)
        axes[3, i].set_ylim(-1, 1)
        axes[3, i].set_aspect('equal', 'box')

    plt.tight_layout()
    plt.savefig(f'generated_trajectory_{img_name}.png')
    plt.close()


# Evaluation function
def evaluate_model(model, dataloader):
    print("IN EVEAL MODEL")
    model.eval()
    batch_index = 0
    with torch.no_grad():
        # batch_idx = 0
        sketches = []
        trajectories = []
        generated_trajectories = []
        fitted_trajectories = []

        for sketch, traj, params, fitted_traj in dataloader:
            sketch = sketch.cuda()

            # Forward pass to get latent variables
            recon, mu, log_var = model(sketch)

            # Generate condition from the latent space
            condition = model.fc_condition(mu)

            # Get the predicted params (from CINN) using the condition
            predicted_params = model.cinn(torch.randn(mu.shape[0], model.n_dim_total).to(mu.device), c=condition, rev=True)[0]

            # Generate trajectory from predicted params
            generated_trajectory = model.generate_trajectory(predicted_params).squeeze()
            generated_trajectories.append(generated_trajectory)
            # print("Generated Trajectories:", generated_trajectories)

            # Append data for visualization
            for i in range(sketch.shape[0]):  # Iterate over each item in the batch
                sketches.append(sketch[i].cpu().numpy().squeeze())
                trajectories.append(traj[i].numpy().squeeze())
                fitted_trajectories.append(fitted_traj[i].numpy().squeeze())

        # Convert lists to numpy arrays for visualization
        sketches = np.array(sketches)
        trajectories = np.array(trajectories)
        fitted_trajectories = np.array(fitted_trajectories)
        generated_trajectories = np.array(generated_trajectories)
        # print("Generated Trajectories AGAIN:", generated_trajectories)

        # Print the shapes for debugging
        # print(f"Sketches shape: {sketches.shape}")
        # print(f"Trajectories shape: {trajectories.shape}")
        # print(f"Generated Trajectories shape: {generated_trajectories.shape}")
        # print(f"Generated trajectory for sample {i}: {generated_trajectories[i]}")

        # Visualize and save the results
        visualize_trajectory(sketches, trajectories, fitted_trajectories, generated_trajectories, img_name=i)
        batch_index += 1


if __name__ == "__main__":
    root_dir = "/fs/nexus-scratch/amishab/Teacher_student_RLsketch/saved_models"

    print("before loading")
    # Initialize the model
    model = VAE_CINN(
        in_channels=1,
        latent_dim=32,
        condition_dim=10,
        num_control_points=10,
        degree=3,    
        encoder=ConvEncoder(in_channels=1, latent_dim=32),
        decoder=ConvDecoder(latent_dim=32, out_channels=1)
    ).cuda()

    print("load state dict")
    # Load the pre-trained model weights
    model.load_state_dict(torch.load('/fs/nexus-scratch/amishab/Teacher_student_RLsketch/saved_models/vae_cinn_finetuning_train/vae_cinn_finetuned_epoch_5.pth'))
    path = '/fs/nexus-scratch/amishab/Teacher_student_RLsketch/saved_models/vae_cinn_finetuning_train/vae_cinn_finetuned_epoch_5.pth'
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Loaded model weights from {path}")
    else:
        raise FileNotFoundError(f"Model file not found at {path}")
    # Get dataloader for evaluation
    dataloader = get_dataloader(batch_size=1, num_samples=6)  # Small batch for visualization

    # Run the evaluation
    evaluate_model(model, dataloader)
