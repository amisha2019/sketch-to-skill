import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from data_func.dataloader_helper import RandomThicken, RandomThinning, Thickening, rescale, normalize, denormalize, standardize, add_noise, log_message


class AugmentedSketchDataset(Dataset):
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

def get_sketch_dataloader(img_size=64, batch_size=32, num_samples=None, val_split=0.2, test_split=0.1, num_workers=8, use_data_aug=True, data_path="new", data_name=None, logger=None):
    """
    Load and preprocess sketch data, apply augmentations, and return DataLoaders.
    
    Args:
        img_size (int): Size of the input images.
        batch_size (int): Batch size for DataLoaders.
        num_samples (int): Number of samples to use (if None, use all).
        val_split (float): Fraction of data to use for validation.
        test_split (float): Fraction of data to use for testing.
        num_workers (int): Number of worker threads for DataLoader.
    
    Returns:
        tuple: Train, validation, and test DataLoaders.
    """
    # Load the data
    if data_path == "old":
        root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/'
        file_names = ["assembly", "boxclose", "coffeepush", "ButtonPress"]
    elif data_path == "new":
        root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new/'
        file_names = ['Dissassemble',
                        'DoorOpen',
                        'DrawerClose',
                        'Hammer',
                        'PegInsertSide',
                        'PickPlace',
                        'PlateSlideBack',
                        'PlateSlideBackSide',
                        'PlateSlideSide',
                        'Push',
                        'PushBack',
                        'PushWall',
                        'Soccer',
                        'StickPush',
                        'Sweep',
                        'SweepInto',
                        'Assembly',
                        'BoxClose',
                        'ButtonPress',
                        'ButtonPressTopdownWall',
                        'CoffeePush',
                        'DrawerOpen',
                        'Reach',
                        'ReachWall',]
        if data_name is not None:
            file_names = [data_name]
    elif data_path == "test":
        root_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/demo_datasets"
        file_names = ['ButtonPress',
                    'ButtonPressTopdownWall',
                    'ButtonPressWall',
                    'CoffeeButton',
                    'DrawerOpen',
                    'Reach',
                    'ReachWall',]
    else:
        raise ValueError("Invalid data path")
        
    sketches1_tensor = []
    sketches2_tensor = []
    for f_name in file_names:
        log_message(f"Loading {f_name}", logger)
        sketches1_tensor.append(torch.load(f'{root_path}/{f_name}/{f_name}_sketches1_{img_size}_cropped.pt'))
        sketches2_tensor.append(torch.load(f'{root_path}/{f_name}/{f_name}_sketches2_{img_size}_cropped.pt'))
    sketches1_tensor = torch.cat(sketches1_tensor, dim=0)
    sketches2_tensor = torch.cat(sketches2_tensor, dim=0)

    # from PIL import Image
    # Image.fromarray(sketch1[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)).save("first_sketch.png")
    # breakpoint()

    # If num_samples is provided, use a subset of the dataset
    if num_samples is not None:
        num_samples = min(num_samples, len(sketches1_tensor))
        indices = np.random.choice(len(sketches1_tensor), num_samples, replace=False)
        sketches1_tensor = sketches1_tensor[indices]
        sketches2_tensor = sketches2_tensor[indices]

    # Combine sketches1 and sketches2 into a single tensor
    combined_sketches_tensor = torch.cat((sketches1_tensor, sketches2_tensor), dim=0)

    # Normalize sketches
    combined_sketches_tensor = normalize(combined_sketches_tensor)
    log_message(f"{len(combined_sketches_tensor)} sketches loaded", logger)
    log_message(f"Raw Sketches - Min: {torch.min(combined_sketches_tensor):.4f}, Max: {torch.max(combined_sketches_tensor):.4f}, Mean: {torch.mean(combined_sketches_tensor):.4f}, Std: {torch.std(combined_sketches_tensor):.4f}", logger)

    # Calculate split sizes
    dataset_size = len(combined_sketches_tensor)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size

    # Shuffle and split indices for train, validation, and test sets
    indices = np.arange(dataset_size)
    # np.random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:val_size + train_size]
    test_indices = indices[val_size + train_size:]

    # if "split" in data_name:
    #     num_stages = 2
    #     nsps = dataset_size // num_stages  # number of samples per stage
    #     val_indices = np.concatenate([np.arange(0, val_size//num_stages), np.arange(nsps, nsps + val_size//num_stages)])   # first val_size//num_stages samples of each stage
    #     train_indices = np.concatenate([np.arange(val_size//num_stages, nsps), np.arange(nsps + val_size//num_stages, dataset_size)])   # last train_size//num_stages samples of each stage
    #     log_message(f"{num_stages} stages, {nsps} samples per stage", logger)
    
    log_message(f"Train indices: {train_indices}, Val indices: {val_indices}, Test indices: {test_indices}", logger)

    # Define data augmentation
    transform_train = transforms.Compose([
        RandomThicken(max_thickness=2),
        # RandomThinning(max_thickness=1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.5, fill=0),  # Add perspective distortion
        transforms.RandomAffine(
            degrees=45,  # Rotation range: -25 to 25 degrees
            translate=(0.2, 0.2),  # Translation range: up to 20% in each direction
            scale=(0.8, 1.5),  # Scaling range: 80% to 150% of original size
            fill=0,
        ),
        # transforms.ElasticTransform(alpha=50.0, sigma=5.0, fill=0),  # Add elastic distortion
        # transforms.Lambda(lambda x: x - 0.5),   # scale to [-0.5, 0.5]
        transforms.Lambda(lambda x: standardize(x)),
        transforms.Lambda(lambda x: add_noise(x)),
        # transforms.Lambda(lambda x: torch.clamp(add_noise(x), -1, 1))  # add noise and clamp
    ]) if use_data_aug else None

    transform_test = transforms.Compose([
        Thickening(thickness=2),
        # transforms.Lambda(lambda x: x - 0.5)   # scale to [-0.5, 0.5]
        transforms.Lambda(lambda x: standardize(x)),
    ]) if use_data_aug else None

    # Create datasets with the original images and the transform
    train_dataset = AugmentedSketchDataset(combined_sketches_tensor[train_indices], transform_train)
    val_dataset = AugmentedSketchDataset(combined_sketches_tensor[val_indices], transform_test)
    test_dataset = AugmentedSketchDataset(combined_sketches_tensor[test_indices], transform_test)

    # Create DataLoader for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, drop_last=True) if train_size > 0 else None
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True) if val_size > 0 else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True) if test_size > 0 else None

    log_message(f"Dataset size: {dataset_size}", logger)
    log_message(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}", logger)

    return train_loader, val_loader, test_loader


# Usage example
if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_sketch_dataloader()

    # Get a batch of data for testing
    for sketches in train_loader:
        print("Sketches shape:", sketches.shape)
        print(f"Max: {sketches.max():.4f}, Min: {sketches.min():.4f}, Mean: {sketches.mean():.4f}, Std: {sketches.std():.4f}")
        
        # Visualize some augmented images
        plt.figure(figsize=(20, 8))
        for i in range(2):
            for j in range(5):
                plt.subplot(2, 5, i*5+j+1)
                print(f"Max: {sketches[i*5+j].max():.4f}, Min: {sketches[i*5+j].min():.4f}, Rescaled Max: {rescale(sketches[i*5+j]).max():.4f}, Rescaled Min: {rescale(sketches[i*5+j]).min():.4f}")

                plt.imshow(rescale(sketches[i*5+j].permute(1, 2, 0)))
                plt.title(f"Augmented Sketch {i*5+j+1}")
                plt.axis('off')
        plt.savefig("augmented_sketches.png")
        
        break  # Only process one batch
