import torch
import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# New dataset: Raw Sketches - Min: 0.0000, Max: 1.0000, Mean: 0.0070, Std: 0.0781
data_mean = 0.0070
data_std = 0.0781
data_max = 22.0

def normalize(data):
    """Normalize the data to [0, 1] range."""
    data = data / 255.0
    # data = (data - data_mean) / data_std / data_max
    return data

def denormalize(data):
    """Denormalize the data back to [0, 255] range."""
    # data = data * data_max
    # data = data * data_std + data_mean
    data = data * 255.0
    return data

def standardize(data):
    """Standardize the data to 0 mean and 1 std."""
    data = (data - data_mean) / data_std
    return data

def add_noise(data):
    """Add noise to the data."""
    noise = torch.randn_like(data) * 0.01
    return data + noise

# def rescale(data):
#     """Rescale the data to [0, 1] range."""
#     min_val = data.min()
#     max_val = data.max()
#     return (data - min_val) / (max_val - min_val)

# def rescale(data):
#     """Rescale the data to [0, 1] range."""
#     data = data * data_std + data_mean
#     if isinstance(data, torch.Tensor):
#         data = data.clamp(0, 1)
#     else:
#         data = np.clip(data, 0, 1)
#     return data

def rescale(data):
    # data = data + 0.5
    # if isinstance(data, torch.Tensor):
    #     data = data.clamp(0, 1)
    # else:
    #     data = np.clip(data, 0, 1)
    data = (data - data.min())/(data.max()-data.min())
    return data

def log_message(message, logger=None):
    if logger:
        logger.log_to_console(message)
    else:
        print(message)

class Thickening(object):
    """Custom transform to thicken lines in the image."""
    def __init__(self, thickness=2):
        self.thickness = thickness

    def __call__(self, img):
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        kernel = np.ones((self.thickness, self.thickness), np.uint8)
        img_np = cv2.dilate(img_np, kernel, iterations=1)
        return torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
    

class Thinning(object):
    """Custom transform to thin lines in the image."""
    def __init__(self, thickness=2):
        self.thickness = thickness

    def __call__(self, img):
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        kernel = np.ones((self.thickness, self.thickness), np.uint8)
        img_np = cv2.erode(img_np, kernel, iterations=1)
        return torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0


class RandomThicken(object):
    """Custom transform to randomly thicken lines in the image."""
    def __init__(self, max_thickness=3):
        self.max_thickness = max_thickness

    def __call__(self, img):
        thickness = random.randint(1, self.max_thickness)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        kernel = np.ones((thickness, thickness), np.uint8)
        img_np = cv2.dilate(img_np, kernel, iterations=1)
        return torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0


class RandomThinning(object):
    """Custom transform to randomly thin lines in the image."""
    def __init__(self, max_thickness=3):
        self.max_thickness = max_thickness

    def __call__(self, img):
        thickness = random.randint(1, self.max_thickness)
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        kernel = np.ones((thickness, thickness), np.uint8)
        img_np = cv2.erode(img_np, kernel, iterations=1)
        return torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0


if __name__ == "__main__":
    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/'
    file_names = ["assembly", "boxclose", "coffeepush", "ButtonPress"]
    sketches1_tensor= torch.load(f'{root_path}/assembly_sketches1_64.pt') / 255.0

    rows = 4
    num_sketches = 6
    sketches1_tensor = sketches1_tensor[:6]
    fig, axs = plt.subplots(rows, num_sketches, figsize=(3*num_sketches, 3*rows))
    for i in range(num_sketches):
        # Original sketch
        original = sketches1_tensor[i].squeeze().permute(1, 2, 0).numpy()
        axs[0, i].imshow(original)
        axs[0, i].set_title(f'Original {i+1}')
        axs[0, i].axis('off')

        # Elastic transform
        elastic_transform = transforms.ElasticTransform(alpha=80.0, sigma=5.0)
        distorted = elastic_transform(sketches1_tensor[i]).squeeze().permute(1, 2, 0).numpy()
        axs[1, i].imshow(distorted)
        axs[1, i].set_title(f'Distorted {i+1}')
        axs[1, i].axis('off')
        
        # Thickened sketch
        thickener = Thickening(thickness=2)
        thickened = thickener(sketches1_tensor[i]).squeeze().permute(1, 2, 0).numpy()
        axs[2, i].imshow(thickened)
        axs[2, i].set_title(f'Thickened {i+1}')
        axs[2, i].axis('off')
        
        # Thinned sketch
        # thinner = Thinning(thickness=2)
        # thinned = thinner(sketches1_tensor[i]).squeeze().permute(1, 2, 0).numpy()
        thinned = (original + distorted) / 2
        axs[3, i].imshow(thinned)
        axs[3, i].set_title(f'Thinned {i+1}')
        axs[3, i].axis('off')

    plt.tight_layout()
    plt.savefig("sketch_transformations.png")
    plt.close()
