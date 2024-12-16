import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from data_func.dataloader import get_dataloader
from model.vae import VAE
from model.vae_mlp import VAE_MLP
from utils.arguments import load_args
from data_func.dataloader_helper import rescale


def interpolate_latent_space(model, test_loader, output_dir, num_interpolations=10):
    model.eval()
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Pick two random batches from the test set
    batch = next(iter(test_loader))
    sketch1, sketch2, traj_gt, params_gt, fitted_traj = batch
    sketch1, sketch2 = sketch1.cuda(), sketch2.cuda()
    
    # Get the latent vectors for both sketches
    with torch.no_grad():
        mu1, _ = model.encoder(sketch1)
        mu2, _ = model.encoder(sketch2)
    
    # Interpolate between the two latent vectors
    interpolated_latents = []
    for alpha in np.linspace(0, 1, num_interpolations):
        interpolated_latents.append((1 - alpha) * mu1 + alpha * mu2)
    interpolated_latents = torch.stack(interpolated_latents, dim=0).cuda()

    # Decode the interpolated latent vectors into 2D sketches
    decoded_sketches = []
    with torch.no_grad():
        for latent in interpolated_latents:
            decoded_sketch = model.decoder(latent).permute(0, 2, 3, 1).cpu().numpy()
            decoded_sketches.append(decoded_sketch)
    decoded_sketches = np.stack(decoded_sketches, axis=0)

    sketch1 = sketch1.permute(0, 2, 3, 1).cpu().numpy()
    sketch2 = sketch2.permute(0, 2, 3, 1).cpu().numpy()

    # Save the interpolated images and generated trajectories
    visualize_and_save_interpolated_images(sketch1, sketch2, decoded_sketches, output_dir)

def visualize_and_save_interpolated_images(original_sketch1, original_sketch2, decoded_sketches, output_dir):
    """
    Save the interpolated 2D sketches and generated trajectories.
    """
    num_interpolations = decoded_sketches.shape[0]
    rows = 6
    cols = num_interpolations + 2
    
    # Total columns should be num_interpolations + 2 (for original sketches)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for idx in range(rows):
        # Plot original sketches
        axes[idx, 0].imshow(rescale(original_sketch1[idx]))
        axes[idx, 0].set_title("Original Sketch 1")

        # Plot interpolated sketches
        for i in range(num_interpolations):
            axes[idx, i + 1].imshow(rescale(decoded_sketches[i][idx]))
            axes[idx, i + 1].set_title(f"Interp {i + 1}")
        
        axes[idx, cols - 1].imshow(rescale(original_sketch2[idx]))
        axes[idx, cols - 1].set_title("Original Sketch 2")

    plt.tight_layout()
    output_file = os.path.join(output_dir, "interpolated_images.png")
    plt.savefig(output_file)
    plt.close()

    print(f"Saved interpolated images and trajectories to {output_file}")


def main(args):
    
    # Load the checkpoint, which contains the model weights directly
    checkpoint = torch.load(args.model_path)
    
    args = load_args(args, args.root_dir)

    # Create the model with the same architecture
    model = VAE(
        img_size=args.img_size, 
        in_channels=3, 
        latent_dim=args.latent_dim, 
        ifPretrained=args.ifPretrained, 
        preTrainedModel_type=args.typeOfPretrainedModel, 
        preTrainedModel_layers=args.layersOfPreTrainedModel, 
        freeze=args.ifFreezePretrainedModel
    ).cuda()
    
    # Load the model's weights directly from the checkpoint
    model.load_state_dict(checkpoint)
    print("Loaded model weights from the checkpoint.")

    # Load the dataset (you only need the test set for interpolation)
    _, _, test_loader = get_dataloader(batch_size=args.bs, num_samples=None, img_size=args.img_size)

    # Interpolate between latent spaces and visualize
    interpolate_latent_space(model, test_loader, args.output_dir, num_interpolations=args.num_interpolations)


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="VAE MLP Latent Space Interpolation Script")
    
    # Provide the model architecture and other dynamic parameters
    parser.add_argument('--img_size', type=int, default=64, help='Size of the input images')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space')
    parser.add_argument('--num_control_points', type=int, default=20, help='Number of control points')
    parser.add_argument('--degree', type=int, default=3, help='Degree of the B-spline')
    
    parser.add_argument('--root_dir', type=str, default='', help='Directory to save the interpolated trajectories')
    parser.add_argument('--model_path', type=str, default='', help='Path to the pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, default='', help='Directory to save the interpolated trajectories')
    parser.add_argument('--num_interpolations', type=int, default=10, help='Number of interpolation points between the two latent vectors')
    parser.add_argument('--bs', type=int, default=256, help='Batch size for loading data')

    parser.add_argument('--ifPretrained', type=bool, default=True, help='if pretrained')
    parser.add_argument('--typeOfPretrainedModel', type=str, default='Resnet', help='type of pretrained model')
    parser.add_argument('--layersOfPreTrainedModel', type=int, default=6, help='number of layers of pretrained model')
    parser.add_argument('--ifFreezePretrainedModel', type=bool, default=True, help='freeze pretrained model')

    args = parser.parse_args()

    args.root_dir = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_Aug_Standardize/vae_with_ResnetPlusOneMlp_6LayerFrozen_2024-09-22_23-45-42_ep200_cosine_lr0.0005_bs256_kld5e-05_aug'
    args.model_path = f'{args.root_dir}/models/vae_model_final.pth'
    args.output_dir = f'{args.root_dir}'

    main(args)
