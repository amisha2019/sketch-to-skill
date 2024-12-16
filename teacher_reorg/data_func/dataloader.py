import torch
import numpy as np
import math
from scipy.interpolate import splprep, splev
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from data_func.dataloader_helper import Thickening, RandomThicken, normalize, standardize, add_noise, log_message


class AugmentedDataset(Dataset):
    """Custom Dataset that applies augmentation on-the-fly"""
    def __init__(self, sketches1, sketches2, trajectories, params, fitted_trajectories, img_transform=None, traj_transform=None):
        self.sketches1 = sketches1
        self.sketches2 = sketches2
        self.trajectories = trajectories
        self.params = params
        self.fitted_trajectories = fitted_trajectories
        self.img_transform = img_transform
        self.traj_transform = traj_transform

    def __len__(self):
        return len(self.sketches1)

    def __getitem__(self, idx):
        sketch1 = self.sketches1[idx]
        sketch2 = self.sketches2[idx]
        trajectory = self.trajectories[idx]
        params = self.params[idx]
        fitted_trajectory = self.fitted_trajectories[idx]

        if self.img_transform:
            sketch1 = self.img_transform(sketch1)
            sketch2 = self.img_transform(sketch2)
        if self.traj_transform:
            trajectory = self.traj_transform(trajectory)
        
        return sketch1, sketch2, trajectory, params, fitted_trajectory


class AugTrajectory(object):
    def __init__(self, noise_scale=0.015, num_control_points=20, smoothness=0.05, p=0.5):
        self.noise_scale = noise_scale
        self.num_control_points = num_control_points
        self.smoothness = smoothness
        self.p = p

    def __call__(self, trajectory):
        if np.random.random() < self.p:
            # Add Gaussian noise
            noisy_trajectory = self.add_gaussian_noise(trajectory)

            # Fit and resample
            augmented_trajectory = self.fit_and_resample_trajectory(noisy_trajectory)
            
            return torch.from_numpy(augmented_trajectory).float()
        else:
            return trajectory

    def add_gaussian_noise(self, trajectory):
        num_points = trajectory.shape[0]
        noise_scale = np.sin(np.linspace(0, np.pi, num_points))
        noise = np.random.normal(0, self.noise_scale, size=trajectory.shape) * noise_scale[:, np.newaxis]
        noise[0] = 0 
        noise[-1] = 0  
        return trajectory + noise

    def fit_and_resample_trajectory(self, trajectory):
        k = 3
        n_knots = self.num_control_points + k + 1
        knots = np.clip(np.linspace(0, 1, n_knots), 0, 1)
        knots[:k+1] = 0
        knots[-k-1:] = 1
        
        # Fit a smoothing spline to the trajectory
        tck, u = splprep(trajectory.T, s=self.smoothness, k=k, t=knots)
        
        # Resample the trajectory with the desired number of control points
        u_new = np.linspace(0, 1, len(trajectory))
        new_trajectory = splev(u_new, tck)
        
        # Keep end points unchanged
        new_trajectory = np.array(new_trajectory).T
        new_trajectory[0] = trajectory[0]
        new_trajectory[-1] = trajectory[-1]
        
        return new_trajectory


def get_dataloader(img_size=64, batch_size=32, num_control_points=20, num_samples=None, val_split=0.2, test_split=0.1, num_workers=8, use_data_aug=True, data_path="new", data_name=None, logger=None):
    # Load the data
    if data_path == "old":
        root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/'
        file_names = ["assembly", "boxclose", "coffeepush", "ButtonPress"]
    elif data_path == "new":
        root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new/'
        if data_name is None:
            file_names = ['Dissassemble',
                        'DoorOpen',
                        'DrawerClose',
                        'Hammer',
                        'PegInsertSide',
                        'PickPlace',
                        'PlateSlideBack',
                        # 'PlateSlideBackSide',
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
        else:
            file_names = [data_name]
    elif data_path == "robomimic":
        root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_robomimic/'
        file_names = [f"{data_name}_part1", f"{data_name}_part2"]
    elif data_path == "test":
        root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/demo_datasets/'
        name_list = ['ButtonPress',
                        'ButtonPressTopdownWall',
                        'DrawerOpen',
                        'Reach',
                        'ReachWall',]
        if data_name is None:
            file_names = name_list
        elif data_name in name_list:
            file_names = [data_name]
        else:
            return None, None, None
    else:
        raise ValueError(f"Invalid data_path: {data_path}")
    
    if data_name is not None and "Real" in data_name:
        root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_real/'
        file_names = [data_name.replace("Real", "")]
        if "toast_pick_place" in data_name:
            file_names = ["bread_pick", "bread_place"]

    log_message(f"Loading from {root_path}", logger)
    
    sketches1_tensor = []
    sketches2_tensor = []
    trajectories_tensor = []
    params_tensor = []
    fitted_trajectories_tensor = []
    for f_name in file_names:
        ff_name = f_name
        suffix = "_cropped"
        if data_path == "robomimic":
            ff_name = f_name.split("_part")[0]
            suffix = ""
        if f_name == "bread_pick" or f_name == "bread_place":
            ff_name = "toast_pick_place"
        if f_name == "ablation":
            suffix = ""
        log_message(f"Loading {f_name}", logger)
        sketches1_tensor.append(torch.load(f'{root_path}/{ff_name}/{f_name}_sketches1_{img_size}{suffix}.pt'))
        sketches2_tensor.append(torch.load(f'{root_path}/{ff_name}/{f_name}_sketches2_{img_size}{suffix}.pt'))
        # trajectories_tensor.append(torch.load(f'{root_path}/{ff_name}/{f_name}_trajectories.pt'))
        trajectories_tensor.append(torch.load(f'{root_path}/{ff_name}/{f_name}_fitted_trajectories_50.pt'))
        params_tensor.append(torch.load(f'{root_path}/{ff_name}/{f_name}_params_{num_control_points}.pt'))
        fitted_trajectories_tensor.append(torch.load(f'{root_path}/{ff_name}/{f_name}_fitted_trajectories_{num_control_points}.pt'))
    sketches1_tensor = torch.cat(sketches1_tensor, dim=0)
    sketches2_tensor = torch.cat(sketches2_tensor, dim=0)
    trajectories_tensor = torch.cat(trajectories_tensor, dim=0)
    params_tensor = torch.cat(params_tensor, dim=0)
    fitted_trajectories_tensor = torch.cat(fitted_trajectories_tensor, dim=0)

    while len(sketches1_tensor) < 64:
        # duplicate the data
        sketches1_tensor = torch.cat([sketches1_tensor, sketches1_tensor], dim=0)
        sketches2_tensor = torch.cat([sketches2_tensor, sketches2_tensor], dim=0)
        trajectories_tensor = torch.cat([trajectories_tensor, trajectories_tensor], dim=0)
        params_tensor = torch.cat([params_tensor, params_tensor], dim=0)
        fitted_trajectories_tensor = torch.cat([fitted_trajectories_tensor, fitted_trajectories_tensor], dim=0)

    log_message(f"Loaded {len(sketches1_tensor)} samples", logger)

    # If num_samples is provided, use a subset of the dataset
    if num_samples is not None:
        num_samples = min(num_samples, len(sketches1_tensor))
        indices = np.random.choice(len(sketches1_tensor), num_samples, replace=False)
        sketches1_tensor = sketches1_tensor[indices]
        sketches2_tensor = sketches2_tensor[indices]
        trajectories_tensor = trajectories_tensor[indices]
        params_tensor = params_tensor[indices]
        fitted_trajectories_tensor = fitted_trajectories_tensor[indices]

    log_message(f"Trajectory max: {[torch.max(trajectories_tensor[:,:,i]).item() for i in range(3)]}, min: {[torch.min(trajectories_tensor[:,:,i]).item() for i in range(3)]}", logger)
    log_message(f"Trajectory x limits: {math.floor(torch.min(trajectories_tensor[:,:,0]).item() * 100) / 100:.2f}, {math.ceil(torch.max(trajectories_tensor[:,:,0]).item() * 100) / 100:.2f}", logger)
    log_message(f"Trajectory y limits: {math.floor(torch.min(trajectories_tensor[:,:,1]).item() * 100) / 100:.2f}, {math.ceil(torch.max(trajectories_tensor[:,:,1]).item() * 100) / 100:.2f}", logger)
    log_message(f"Trajectory z limits: {math.floor(torch.min(trajectories_tensor[:,:,2]).item() * 100) / 100:.2f}, {math.ceil(torch.max(trajectories_tensor[:,:,2]).item() * 100) / 100:.2f}", logger)
    log_message(f"Fitted trajectory max: {[torch.max(fitted_trajectories_tensor[:,:,i]).item() for i in range(3)]}, min: {[torch.min(fitted_trajectories_tensor[:,:,i]).item() for i in range(3)]}", logger)
    log_message(f"Fitted trajectory x limits: {math.floor(torch.min(fitted_trajectories_tensor[:,:,0]).item() * 100) / 100:.2f}, {math.ceil(torch.max(fitted_trajectories_tensor[:,:,0]).item() * 100) / 100:.2f}", logger)
    log_message(f"Fitted trajectory y limits: {math.floor(torch.min(fitted_trajectories_tensor[:,:,1]).item() * 100) / 100:.2f}, {math.ceil(torch.max(fitted_trajectories_tensor[:,:,1]).item() * 100) / 100:.2f}", logger)
    log_message(f"Fitted trajectory z limits: {math.floor(torch.min(fitted_trajectories_tensor[:,:,2]).item() * 100) / 100:.2f}, {math.ceil(torch.max(fitted_trajectories_tensor[:,:,2]).item() * 100) / 100:.2f}", logger)

    # Normalize sketches from [0, 255] to [0, 1]
    sketches1_tensor = normalize(sketches1_tensor)
    sketches2_tensor = normalize(sketches2_tensor)

    # Print the statistics of the combined sketches
    combined_sketches_tensor = torch.cat((sketches1_tensor, sketches2_tensor), dim=0)
    log_message(f"{len(sketches1_tensor)} samples loaded", logger)
    log_message(f"Raw Sketches - Min: {torch.min(combined_sketches_tensor):.4f}, Max: {torch.max(combined_sketches_tensor):.4f}, Mean: {torch.mean(combined_sketches_tensor):.4f}, Std: {torch.std(combined_sketches_tensor):.4f}", logger)

    # Calculate split sizes
    dataset_size = len(sketches1_tensor)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # Shuffle and split indices for train, validation, and test sets
    indices = np.arange(dataset_size)
    # np.random.shuffle(indices)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:val_size + train_size]
    test_indices = indices[val_size + train_size:]

    if data_path == "robomimic":
        num_stages = len(file_names)
        nsps = dataset_size // num_stages  # number of samples per stage
        val_indices = np.concatenate([np.arange(0, val_size//num_stages), np.arange(nsps, nsps + val_size//num_stages)])   # first val_size//num_stages samples of each stage
        train_indices = np.concatenate([np.arange(val_size//num_stages, nsps), np.arange(nsps + val_size//num_stages, dataset_size)])   # last train_size//num_stages samples of each stage
        log_message(f"use first {val_size//num_stages} samples of each stage for validation", logger)
    if "split" in data_name:
        num_stages = 2
        nsps = dataset_size // num_stages  # number of samples per stage
        val_indices = np.concatenate([np.arange(0, val_size//num_stages), np.arange(nsps, nsps + val_size//num_stages)])   # first val_size//num_stages samples of each stage
        train_indices = np.concatenate([np.arange(val_size//num_stages, nsps), np.arange(nsps + val_size//num_stages, dataset_size)])   # last train_size//num_stages samples of each stage
        log_message(f"{num_stages} stages, {nsps} samples per stage", logger)
    
    log_message(f"Train indices: {train_indices}, Val indices: {val_indices}, Test indices: {test_indices}", logger)

    # if data_name == "Assembly_gradient":
    #     val_indices = np.arange(0, val_size)
    #     train_indices = np.arange(val_size, dataset_size)
    #     log_message(f"dataset size: {dataset_size}, val size: {val_size}, train size: {train_size}", logger)
    #     log_message(f"Train indices: {train_indices}, Val indices: {val_indices}", logger)
    
    transform_train = transforms.Compose([
        RandomThicken(max_thickness=2),
        transforms.ElasticTransform(alpha=80.0, sigma=5.0),
        # transforms.Lambda(lambda x: x - 0.5),   # scale to [-0.5, 0.5]
        transforms.Lambda(lambda x: standardize(x)),
        transforms.Lambda(lambda x: add_noise(x)),
        # transforms.Lambda(lambda x: torch.clamp(add_noise(x), -1, 1))  # add noise
    ]) if use_data_aug else None

    transform_test = transforms.Compose([
        Thickening(thickness=2),
        transforms.Lambda(lambda x: standardize(x)),
        # transforms.Lambda(lambda x: x - 0.5),   # scale to [-0.5, 0.5]
    ]) if use_data_aug else None

    transform_traj = AugTrajectory(noise_scale=0.015, num_control_points=20, smoothness=0.05, p=0.6) if use_data_aug else None
    
    # Create datasets
    datasets = []
    for indices, transform in [(train_indices, transform_train), (val_indices, transform_test), (test_indices, transform_test)]:
        datasets.append(AugmentedDataset(
            *[tensor[indices] for tensor in (sketches1_tensor, sketches2_tensor, trajectories_tensor, params_tensor, fitted_trajectories_tensor)],
            img_transform=transform,
            traj_transform=transform_traj if transform == transform_train else None,
            # traj_transform=None,
        ))
    train_dataset, val_dataset, test_dataset = datasets

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


if __name__ == '__main__':
    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets'
    file_names = ["assembly", "boxclose", "coffeepush", "ButtonPress"]

    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_real'
    f_name = "toast_press"
    trajectories = torch.load(f'{root_path}/{f_name}/{f_name}_fitted_trajectories_50.pt')

    num_subplot = 2
    num_samples = 6
    num_random = 5
    fig = plt.figure(figsize=(3*num_samples, 3*num_subplot))
    aug_traj = AugTrajectory(noise_scale=0.005, num_control_points=10, smoothness=0.005, p=1)
    for i in range(num_samples):
        raw_traj = trajectories[i]
        traj = raw_traj.numpy()
        ax1 = fig.add_subplot(num_subplot, num_samples, i+1, projection='3d')
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], 'b-', linewidth=2)
        ax1.set_title(f"Original Trajectory {i+1}")

        ax2 = fig.add_subplot(num_subplot, num_samples, num_samples + i + 1, projection='3d')
        for j in range(num_random):
            augmented_traj = aug_traj(raw_traj).numpy()
            ax2.plot(augmented_traj[:, 0], augmented_traj[:, 1], augmented_traj[:, 2], '-', linewidth=2)
            ax2.set_title(f"Augmented Trajectory {i+1}")

    plt.tight_layout()
    plt.savefig(f"augmented_trajectories_{f_name}.png")
    plt.close()
