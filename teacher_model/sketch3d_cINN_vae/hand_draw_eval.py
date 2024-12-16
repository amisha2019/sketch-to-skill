import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import VAE_MLP
from PIL import Image

def normalize_sketch(sketch):
    if sketch.max() > 1.0:
        sketch = sketch / 255.0
    data_mean = 0.0037
    data_std = 0.0472
    data_max = 22.0
    return (sketch - data_mean) / (data_max * data_std)

def unnormalize_sketch(sketch):
    data_mean = 0.0037
    data_std = 0.0472
    data_max = 22.0
    return sketch * data_max * data_std + data_mean

def visualize_trajectory(sketches1, sketches2, reconstructed_sketches1, reconstructed_sketches2, generated_trajectories, img_name):
    num_samples = len(sketches1)
    rows = 5
    fig, axes = plt.subplots(figsize=(6*num_samples, 6*rows))

    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    for i in range(num_samples):
        ax1 = fig.add_subplot(rows, num_samples, i+1)
        ax1.imshow(np.flipud(unnormalize_sketch(sketches1[i].squeeze())))
        ax1.set_title(f"Sample {i+1}: Sketch 1")
        ax1.set_aspect('equal', 'box')

        ax2 = fig.add_subplot(rows, num_samples, num_samples+i+1)
        ax2.imshow(np.flipud(unnormalize_sketch(sketches2[i].squeeze())))
        ax2.set_title(f"Sample {i+1}: Sketch 2")
        ax2.set_aspect('equal', 'box')

        ax3 = fig.add_subplot(rows, num_samples, 2*num_samples+i+1)
        ax3.imshow(np.flipud(unnormalize_sketch(reconstructed_sketches1[i].squeeze())))
        ax3.set_title(f"Sample {i+1}: Reconstructed Sketch 1")
        ax3.set_aspect('equal', 'box')

        ax4 = fig.add_subplot(rows, num_samples, 3*num_samples+i+1)
        ax4.imshow(np.flipud(unnormalize_sketch(reconstructed_sketches2[i].squeeze())))
        ax4.set_title(f"Sample {i+1}: Reconstructed Sketch 2")
        ax4.set_aspect('equal', 'box')

        ax6 = fig.add_subplot(rows, num_samples, 4*num_samples+i+1, projection='3d')
        ax6.plot(generated_trajectories[i][:, 0], generated_trajectories[i][:, 1], generated_trajectories[i][:, 2])
        ax6.set_xlabel('X')
        ax6.set_ylabel('Y')
        ax6.set_zlabel('Z')
        ax6.set_title(f"Sample {i+1}: Generated Trajectory")
        
    plt.tight_layout()
    plt.savefig(f'eval_results/hand_draw_generated_trajectory_{img_name}.png')

from data_vae import get_sketch_dataloader, normalize, rescale, Thickening
thickening = Thickening(thickness=4)

def evaluate_model(model, sketches):
    model.eval()
    with torch.no_grad():
        i = 0
        for sketch1, sketch2 in sketches:
            sketch1 = normalize(sketch1)
            sketch2 = normalize(sketch2)
            # Apply thickening to the sketch
            sketch1 = thickening(torch.from_numpy(sketch1).permute(2, 0, 1))
            sketch2 = thickening(torch.from_numpy(sketch2).permute(2, 0, 1))
            sketch1 = sketch1.permute(1, 2, 0).numpy()
            sketch2 = sketch2.permute(1, 2, 0).numpy()

            # Resize the sketch to match the desired image size
            sketch1 = torch.nn.functional.interpolate(
                torch.from_numpy(sketch1).unsqueeze(0).permute(0, 3, 1, 2),
                size=(img_size, img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0).numpy()
            sketch2 = torch.nn.functional.interpolate(
                torch.from_numpy(sketch2).unsqueeze(0).permute(0, 3, 1, 2),
                size=(img_size, img_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0).permute(1, 2, 0).numpy()

            sketch1 = torch.tensor(sketch1).unsqueeze(0).permute(0, 3, 1, 2).float()
            sketch2 = torch.tensor(sketch2).unsqueeze(0).permute(0, 3, 1, 2).float()
            sketch1 += 0.005 * torch.randn_like(sketch1)
            sketch2 += 0.005 * torch.randn_like(sketch2)

            sketch1 = sketch1.cuda()
            sketch2 = sketch2.cuda()

            sketches1 = []
            sketches2 = []
            reconstructed_sketches1 = []
            reconstructed_sketches2 = []
            generated_trajectories = []

            for _ in range(6):
                recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params = model(sketch1, sketch2)
                print(params)

                generated_trajectory = model.generate_trajectory(params)
                # traj = model.bspline_curve(params)
                # print(torch.abs(traj - generated_trajectory).max(), torch.abs(traj - generated_trajectory).min())
                # tensor(3.9334e-07, device='cuda:0', dtype=torch.float64) tensor(0., device='cuda:0', dtype=torch.float64)
                # equivalent

                generated_trajectories.append(generated_trajectory.cpu().numpy().squeeze())

                sketches1.append(sketch1.cpu().numpy().squeeze().transpose(1, 2, 0))
                sketches2.append(sketch2.cpu().numpy().squeeze().transpose(1, 2, 0))
                reconstructed_sketches1.append(recons1.cpu().numpy().squeeze().transpose(1, 2, 0))
                reconstructed_sketches2.append(recons2.cpu().numpy().squeeze().transpose(1, 2, 0))

            
            visualize_trajectory(sketches1, sketches2, reconstructed_sketches1, reconstructed_sketches2, generated_trajectories, img_name=i)
            i += 1

if __name__ == "__main__":
    model_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_3D/vae_mlp_2024-09-20_04-47-25_lr1e-3_equalWeight4Loss_Euclidean/bspline_model_best.pth"

    img_size = 64
    num_control_points = 20
    model = VAE_MLP(img_size=img_size,
                    in_channels=3,
                    latent_dim=256,
                    num_control_points=num_control_points,
                    degree=3,).cuda()
    model.load_state_dict(torch.load(model_path))
    
    sketches_names = [
        ["rgb_images/demo_0_corner_sketch.tiff", "rgb_images/demo_0_corner2_sketch.tiff"],
        ["rgb_images/demo_6_corner_sketch.tiff", "rgb_images/demo_6_corner2_sketch.tiff"],
    ]
    # load the sketches
    # sketches = [[np.flipud(np.array(Image.open(sketch).convert('RGB'))) for sketch in sketches_names[i]] for i in range(2)]
    sketches = [[np.array(Image.open(sketch).convert('RGB')) for sketch in sketches_names[i]] for i in range(2)]
    # sketches = np.array([np.array([np.array(Image.open(sketch).convert('RGB')) for sketch in sketches_names[i]]) for i in range(2)])
    evaluate_model(model, sketches)
