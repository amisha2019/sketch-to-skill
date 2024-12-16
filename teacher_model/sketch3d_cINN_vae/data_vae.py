import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset, Dataset  # Add Dataset here
import torchvision.transforms as transforms
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

data_mean = 0.0037
data_std = 0.0472
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

# def rescale(data):
#     """Rescale the data to [0, 1] range."""
#     min_val = data.min()
#     max_val = data.max()
#     return (data - min_val) / (max_val - min_val)    

def rescale(data):
    data = data + 0.5
    if isinstance(data, torch.Tensor):
        data = data.clamp(0, 1)
    else:
        data = np.clip(data, 0, 1)
    return data

class Thickening(object):
    """Custom transform to thicken lines in the image."""
    def __init__(self, thickness=2):
        self.thickness = thickness

    def __call__(self, img):
        img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        kernel = np.ones((self.thickness, self.thickness), np.uint8)
        img_np = cv2.dilate(img_np, kernel, iterations=1)
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

def visualize_augmentations(original_img, transform, num_augmentations=5):
    """Visualize the original image and its augmentations."""
    fig, axs = plt.subplots(1, num_augmentations + 1, figsize=(20, 4))
    axs[0].imshow(original_img.permute(1, 2, 0))
    axs[0].set_title('Original')
    for i in range(num_augmentations):
        aug_img = transform(original_img)
        axs[i+1].imshow(aug_img.permute(1, 2, 0))
        axs[i+1].set_title(f'Augmentation {i+1}')
    plt.show()

class AugmentedDataset(Dataset):
    """Custom Dataset that applies augmentation on-the-fly"""
    def __init__(self, tensor, transform=None):
        self.tensor = tensor
        self.transform = transform

    def __len__(self):
        return len(self.tensor)

    def __getitem__(self, idx):
        image = self.tensor[idx]
        if self.transform:
            image = self.transform(image)
        return image

def get_sketch_dataloader(img_size=64, batch_size=32, num_control_points=20, num_samples=None, val_split=0.2, test_split=0.1, num_workers=8):
    """
    Load and preprocess sketch data, apply augmentations, and return DataLoaders.
    
    Args:
        img_size (int): Size of the input images.
        batch_size (int): Batch size for DataLoaders.
        num_control_points (int): Number of control points for the sketches.
        num_samples (int): Number of samples to use (if None, use all).
        val_split (float): Fraction of data to use for validation.
        test_split (float): Fraction of data to use for testing.
        num_workers (int): Number of worker threads for DataLoader.
    
    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    # Load the data
    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/'
    sketches1_tensor = torch.load(root_path + f'sketches1_{img_size}.pt')
    sketches2_tensor = torch.load(root_path + f'sketches2_{img_size}.pt')

    # If num_samples is provided, use a subset of the dataset
    if num_samples is not None:
        num_samples = min(num_samples, len(sketches1_tensor))
        indices = np.random.choice(len(sketches1_tensor), num_samples, replace=False)
        sketches1_tensor = sketches1_tensor[indices]
        sketches2_tensor = sketches2_tensor[indices]

    # Normalize sketches
    sketches1_tensor = normalize(sketches1_tensor)
    sketches2_tensor = normalize(sketches2_tensor)

    print(f"Sketch 1 - Min: {torch.min(sketches1_tensor):.4f}, Max: {torch.max(sketches1_tensor):.4f}, Mean: {torch.mean(sketches1_tensor):.4f}, Std: {torch.std(sketches1_tensor):.4f}")
    print(f"Sketch 2 - Min: {torch.min(sketches2_tensor):.4f}, Max: {torch.max(sketches2_tensor):.4f}, Mean: {torch.mean(sketches2_tensor):.4f}, Std: {torch.std(sketches2_tensor):.4f}")

    # Combine sketches1 and sketches2 into a single tensor
    combined_sketches_tensor = torch.cat((sketches1_tensor, sketches2_tensor), dim=0)

    # Calculate split sizes
    dataset_size = len(combined_sketches_tensor)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size

    # Shuffle and split indices for train, validation, and test sets
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    # Define data augmentation
    transform = transforms.Compose([
        RandomThicken(max_thickness=2),
        RandomThinning(max_thickness=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(
            degrees=15,  # Rotation range: -15 to 15 degrees
            translate=(0.2, 0.2),  # Translation range: up to 20% in each direction
            scale=(0.8, 1.5),  # Scaling range: 80% to 150% of original size
        ),
        transforms.RandomPerspective(distortion_scale=0.2, p=1),  # Add perspective distortion
        transforms.ElasticTransform(alpha=50.0, sigma=5.0),  # Add elastic distortion
        transforms.Lambda(lambda x: torch.clamp(x + 0.01 * torch.randn_like(x) - 0.5, -1, 1))
    ])

    transform_test = transforms.Compose([
        Thickening(thickness=2),
        transforms.Lambda(lambda x: x - 0.5)
    ])

    # Create datasets with the original images and the transform
    train_dataset = AugmentedDataset(combined_sketches_tensor[train_indices], transform)
    val_dataset = AugmentedDataset(combined_sketches_tensor[val_indices], transform_test)
    # test_dataset = AugmentedDataset(combined_sketches_tensor[test_indices], transform)
    test_dataset = AugmentedDataset(combined_sketches_tensor[test_indices], transform_test)

    # Create DataLoader for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True)

    print(f"Dataset size: {dataset_size}")
    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

    return train_loader, val_loader, test_loader

# Usage example
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_sketch_dataloader(num_samples=64)

    # Get a batch of data for testing
    for batch in train_loader:
        sketches = batch
        # breakpoint()
        print("Sketches shape:", sketches.shape)
        
        # Visualize some augmented images
        plt.figure(figsize=(20, 4))
        for i in range(5):
            plt.subplot(1, 5, i+1)
            print(f"Max: {sketches[i].max():.4f}, Min: {sketches[i].min():.4f}")
            plt.imshow(rescale(sketches[i].permute(1, 2, 0)))
            # plt.imshow(sketches[i].permute(1, 2, 0))
            plt.title(f"Augmented Sketch {i+1}")
            plt.axis('off')
        plt.savefig("augmented_sketches.png")
        
        break  # Only process one batch