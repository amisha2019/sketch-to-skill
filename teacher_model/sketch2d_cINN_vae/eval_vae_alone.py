import numpy as np
import torch
import matplotlib.pyplot as plt
from data import get_dataloader
from model import VAE

def visualize_sketch(sketches, reconstructed_sketches, img_name=None):
    num_samples = len(sketches)
    fig, axes = plt.subplots(2, num_samples, figsize=(3*num_samples, 6))
    
    for i in range(num_samples):
        axes[0, i].imshow(np.flipud(sketches[i].squeeze()), cmap='gray')
        axes[0, i].set_title(f"Sample {i+1}: Sketch")
        axes[1, i].set_aspect('equal', 'box')

        axes[1, i].imshow(np.flipud(reconstructed_sketches[i].squeeze()), cmap='gray')
        axes[1, i].set_title(f"Sample {i+1}: Reconstructed Sketch")
        axes[1, i].set_aspect('equal', 'box')

    plt.tight_layout()
    plt.savefig(f'vae_results{img_name}.png')
    

def evaluate_model(model, dataloader):
    model.eval()
    with torch.no_grad():
        sketches = []
        reconstructed_sketches = []
        for sketch, _, _, _ in dataloader:
            sketch = sketch.cuda()
            recons, input, mu, log_var = model(sketch)
            sketches.append(sketch.cpu().numpy().squeeze())
            reconstructed_sketches.append(recons.cpu().numpy().squeeze())

        visualize_sketch(sketches, reconstructed_sketches, img_name="_vae_alone")


if __name__ == "__main__":
    img_size = 64
    root_dir = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong"
    model = VAE(img_size=img_size, in_channels=1, latent_dim=128).cuda()
    model.load_state_dict(torch.load(f'{root_dir}/vae_model.pth'))

    dataloader = get_dataloader(batch_size=1, num_samples=6, img_size=img_size)
    evaluate_model(model, dataloader)
