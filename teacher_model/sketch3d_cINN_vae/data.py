import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import interpolate
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
import h5py
import time
# from chamferdist import ChamferDistance
from distances import ChamferDistance, FrechetDistance
from torch.utils.data import Subset
import torchvision.transforms as transforms
from data_vae import Thickening, RandomThicken, RandomThinning


def bspline_basis(i, p, t, knots):
    """
    Compute the i-th B-spline basis function of degree p at parameter t.
    """
    if p == 0:
        return 1.0 if knots[i] <= t < knots[i+1] else 0.0
    else:
        w1 = (t - knots[i]) / (knots[i+p] - knots[i]) if knots[i+p] != knots[i] else 0.0
        w2 = (knots[i+p+1] - t) / (knots[i+p+1] - knots[i+1]) if knots[i+p+1] != knots[i+1] else 0.0
        return w1 * bspline_basis(i, p-1, t, knots) + w2 * bspline_basis(i+1, p-1, t, knots)

def bspline_curve(control_points, knots, degree, num_points):
    """
    Generate points on a B-spline curve.
    
    Parameters:
    control_points: list of numpy arrays representing the control points.
    knots: list of knots.
    degree: degree of the B-spline.
    num_points: number of points to generate on the curve.
    
    Returns:
    list of numpy arrays representing points on the B-spline curve.
    """
    n = len(control_points) - 1
    t_values = np.linspace(knots[degree], knots[n+1], num_points)
    basis_matrix = np.array([[bspline_basis(i, degree, t, knots) for i in range(n+1)] for t in t_values])
    basis_matrix[-1, -1] = 1.0  # ensure the last point is included
    points = np.dot(basis_matrix, control_points)
    
    return points

def get_chamfer_distance(points1, points2):
    chamferDist = ChamferDistance()
    if not isinstance(points1, torch.Tensor):
        points1 = torch.from_numpy(points1)
    if not isinstance(points2, torch.Tensor):
        points2 = torch.from_numpy(points2)
    points1 = points1.unsqueeze(0).float()
    points2 = points2.unsqueeze(0).float()
    # dist = chamferDist(points1, points2, bidirectional=True)
    dist = chamferDist(points1, points2).sum()
    return dist

def get_frechet_distance(points1, points2):
    frechetDist = FrechetDistance()
    if not isinstance(points1, torch.Tensor):
        points1 = torch.from_numpy(points1)
    if not isinstance(points2, torch.Tensor):
        points2 = torch.from_numpy(points2)
    points1 = points1.unsqueeze(0).float()
    points2 = points2.unsqueeze(0).float()
    dist = frechetDist(points1, points2).sum()
    return dist


def fit_trajectory_bspline(points, num_control_points=20, s=0, k=3):
    u = np.linspace(0, 1, len(points))
    n_knots = num_control_points + k + 1
    knots = np.clip(np.linspace(0, 1, n_knots), 0, 1)
    knots[:k+1] = 0
    knots[-k-1:] = 1
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    if points.shape[1] == 2:
        tck, u = splprep([points[:, 0], points[:, 1]], u=u, s=s, k=k, t=knots, task=-1)
    else:
        tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], u=u, s=s, k=k, t=knots, task=-1)
    return tck, u

def predict_trajectory_bspline(tck, num_points=200):
    u_new = np.linspace(0, 1, num_points)
    return np.array(splev(u_new, tck)).T

def get_trajectory_params_bspline(trajectories, num_control_points=10):
    params = []
    fitted_trajectories = []

    for i, points in enumerate(trajectories):
        tck, u = fit_trajectory_bspline(points, num_control_points)
        # tck2, u = fit_trajectory_bspline(points, 50)
        # breakpoint()
        # ct1 = np.array(tck[1]).T
        # ct2 = np.array(tck2[1]).T
        # ft1 = predict_trajectory_bspline(tck, num_points=200)
        # ft2 = predict_trajectory_bspline(tck2, num_points=200)
        # print(np.mean(np.abs(ft1 - ft2)))
        # 0.003135697486369386, not very small

        ##### method from scipy:
        fitted_trajectory = predict_trajectory_bspline(tck, num_points=100)
        ##### implementation 1:
        # control_points = np.array(tck[1]).T
        # fitted_trajectory = bspline_curve(control_points, tck[0], tck[2], 100)
        # fitted_trajectory = np.array(fitted_trajectory)
        # print(np.array_equal(fitted_trajectory_raw, fitted_trajectory))
        # print(np.mean(np.abs(fitted_trajectory_raw - fitted_trajectory)))
        ##### implementation 2:
        # control_points = torch.from_numpy(np.array(tck[1]).T).unsqueeze(0).float()
        # fitted_trajectory = bspline_interpolation(control_points, num_points=200, degree=3)
        # fitted_trajectory = fitted_trajectory.squeeze().numpy()
        # breakpoint()
        # print(f"Range of fitted coefficients: {np.min(tck[1])} to {np.max(tck[1])}")

        # # points = torch.from_numpy(points).unsqueeze(0).float()
        # # fitted_trajectory = torch.from_numpy(fitted_trajectory).unsqueeze(0).float()
        # chamfer_dist = get_chamfer_distance(points, fitted_trajectory)
        # print(f"Chamfer distance: {chamfer_dist}")
        # frechet_dit = get_frechet_distance(points, fitted_trajectory)
        # print(f"Frechet distance: {frechet_dit}\n")

        params.append(np.array(tck[1]).flatten())
        fitted_trajectories.append(fitted_trajectory)
        print(f"Trajectory {i}/{trajectories.shape[0]} fitted.")

    return np.array(params), np.array(fitted_trajectories)

def basis_functions(s, num_basis=10):
    """Compute basis functions for given s values."""
    ti = np.linspace(0, 1, num_basis)
    # smaller values correspond to smoother trajectories
    return np.exp(-20 * (s[:, np.newaxis] - ti)**2)

def trajectory_model(s, weights, num_basis=10):
    """Compute trajectory points for given s values and weights."""
    phi = basis_functions(s, num_basis)
    # breakpoint()
    return phi @ weights

def objective_function(weights, points, num_basis, num_reference_points=200):
    """Compute the error between the model and the actual points."""
    weights = weights.reshape(-1, 3)
    s = np.linspace(0, 1, len(points))
    model_points = trajectory_model(s, weights, num_basis)
    distance = np.sum((model_points - points)**2)
    
    # s = np.linspace(0, 1, num_reference_points)
    # model_points = trajectory_model(s, weights, num_basis)
    # chamferDist = ChamferDistance()
    # points = torch.from_numpy(points).unsqueeze(0).float()
    # model_points = torch.from_numpy(model_points).unsqueeze(0).float()
    # dist_bidirectional = chamferDist(points, model_points, bidirectional=True)
    return distance

def fit_trajectory_seb(points, num_basis=10):
    # Initialize weights randomly
    initial_weights = np.random.randn(num_basis, 3)
    
    # Optimize weights
    result = minimize(
        objective_function, 
        initial_weights.flatten(), 
        args=(points, num_basis),
        method='L-BFGS-B'
    )
    
    return result.x.reshape(-1, 3)

def predict_trajectory_seb(weights, num_points=200, num_basis=10):
    s_smooth = np.linspace(0, 1, num_points)
    fitted_trajectory = trajectory_model(s_smooth, weights, num_basis)
    return fitted_trajectory

def get_trajectory_params_seb(trajectories, num_basis=10):
    params = []
    fitted_trajectories = []

    for points in trajectories:
        tck, u = fit_trajectory_bspline(points, num_control_points=40)
        new_points = predict_trajectory_bspline(tck, num_points=400)
        dist_bidirectional = get_chamfer_distance(points, new_points)
        print(f"Chamfer distanc of Bspline: {dist_bidirectional}")

        weights = fit_trajectory_seb(new_points, num_basis)
        print(f"Range of fitted weights: {np.min(weights)} to {np.max(weights)}")
        fitted_trajectory = predict_trajectory_seb(weights, num_points=200, num_basis=num_basis)

        dist_bidirectional = get_chamfer_distance(points, fitted_trajectory)
        print(f"Chamfer distance: {dist_bidirectional}\n")

        params.append(weights.flatten())
        fitted_trajectories.append(fitted_trajectory)

    return np.array(params), np.array(fitted_trajectories)

def load_data(file_path=None, num_samples=None, img_size=None):
    if file_path is None:
        file_path = ['/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_datasets/sketch_data_assembly.hdf5', 
                     '/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_datasets/sketch_data_boxclose.hdf5',
                     '/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_datasets/sketch_data_coffeepush.hdf5',]
    sketches1 = []
    sketches2 = []
    trajectories = []

    # Load the dataset
    for cur_file_path in file_path:
        with h5py.File(cur_file_path, 'r') as f:
            print(f"Loading data from {cur_file_path}")
            length = len(list(f.keys()))
            index = np.arange(length)
            if num_samples is not None:
                index = np.random.choice(length, num_samples//len(file_path), replace=False)

            for i in index:
                sketches1.append(f[f'demo_{i}/corner'][:])
                sketches2.append(f[f'demo_{i}/corner2'][:])
                trajectories.append(f[f'demo_{i}/obs'][:, :3])

    sketches1 = np.array(sketches1)
    sketches2 = np.array(sketches2)
    if img_size is not None:
        sketches1 = np.array([np.array(Image.fromarray(sketch).resize((img_size, img_size))) for sketch in sketches1])
        sketches2 = np.array([np.array(Image.fromarray(sketch).resize((img_size, img_size))) for sketch in sketches2])
    trajectories = np.array(trajectories)
    print(f"Loaded {len(sketches1)} images")
    print(f"Image shape: {sketches1.shape}")
    return sketches1, sketches2, trajectories


def get_dataloader_hdf5(img_size=64, batch_size=32, num_control_points=20, num_samples=None):
    sketches1, sketches2, trajectories = load_data(num_samples=num_samples, img_size=img_size)
    sketches1_tensor = torch.from_numpy(sketches1).permute(0, 3, 1, 2).float() / 255.0
    sketches2_tensor = torch.from_numpy(sketches2).permute(0, 3, 1, 2).float() / 255.0
    print(f"Dataset generated with {len(sketches1_tensor)} samples.")
    print(f"Sketch shape: {sketches1_tensor.shape}")
    print(f"Sketch min: {sketches1_tensor.min()}", f"Sketch max: {sketches1_tensor.max()}")

    params, fitted_trajectories = get_trajectory_params_bspline(trajectories, num_control_points=num_control_points)
    trajectories_tensor = torch.from_numpy(trajectories).float()
    params_tensor = torch.from_numpy(params).float()
    fitted_trajectories_tensor = torch.from_numpy(fitted_trajectories).float()

    dataset = TensorDataset(sketches1_tensor, sketches2_tensor, trajectories_tensor, params_tensor, fitted_trajectories_tensor)
    
    print(f"Trajectory shape: {trajectories_tensor.shape}")
    print(f"Params shape: {params_tensor.shape}")
    print(f"Fitted Trajectory shape: {fitted_trajectories.shape}")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
            fitted_trajectory = self.traj_transform(fitted_trajectory)
        
        return sketch1, sketch2, trajectory, params, fitted_trajectory

class AugTrajectory(object):
    def __init__(self, noise_scale=0.015, num_control_points=20, smoothness=0.05):
        self.noise_scale = noise_scale
        self.num_control_points = num_control_points
        self.smoothness = smoothness

    def __call__(self, trajectory):
        # Add Gaussian noise
        noisy_trajectory = self.add_gaussian_noise(trajectory)
        
        # Fit and resample
        augmented_trajectory = self.fit_and_resample_trajectory(noisy_trajectory)
        
        return torch.from_numpy(augmented_trajectory).float()

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
        

def get_dataloader(img_size=64, batch_size=32, num_control_points=20, num_samples=None, val_split=0.2, test_split=0.1):
    # Load the data
    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/'
    sketches1_tensor = torch.load(root_path + f'sketches1_{img_size}.pt')
    sketches2_tensor = torch.load(root_path + f'sketches2_{img_size}.pt')
    trajectories_tensor = torch.load(root_path + 'trajectories.pt')
    params_tensor = torch.load(root_path + f'params_{num_control_points}.pt')
    fitted_trajectories_tensor = torch.load(root_path + f'fitted_trajectories_50.pt')

    # If num_samples is provided, use a subset of the dataset
    if num_samples is not None:
        num_samples = min(num_samples, len(sketches1_tensor))
        indices = np.random.choice(len(sketches1_tensor), num_samples, replace=False)
        sketches1_tensor = sketches1_tensor[indices]
        sketches2_tensor = sketches2_tensor[indices]
        trajectories_tensor = trajectories_tensor[indices]
        params_tensor = params_tensor[indices]
        fitted_trajectories_tensor = fitted_trajectories_tensor[indices]

    # Normalize sketches
    sketches1_tensor = sketches1_tensor / 255.0
    sketches2_tensor = sketches2_tensor / 255.0
    
    # Calculate split sizes
    dataset_size = len(sketches1_tensor)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    # Shuffle and split indices for train, validation, and test sets
    indices = np.arange(dataset_size)
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    transform_train = transforms.Compose([
        RandomThicken(max_thickness=2),
    ])

    transform_test = transforms.Compose([
        Thickening(thickness=2),
    ])

    transform_traj = AugTrajectory(noise_scale=0.015, num_control_points=20, smoothness=0.05)
    
    # Create subsets
    train_dataset = AugmentedDataset(sketches1_tensor[train_indices], sketches2_tensor[train_indices], trajectories_tensor[train_indices], params_tensor[train_indices], fitted_trajectories_tensor[train_indices], img_transform=transform_train, traj_transform=transform_traj)
    val_dataset = AugmentedDataset(sketches1_tensor[val_indices], sketches2_tensor[val_indices], trajectories_tensor[val_indices], params_tensor[val_indices], fitted_trajectories_tensor[val_indices], img_transform=transform_test)
    test_dataset = AugmentedDataset(sketches1_tensor[test_indices], sketches2_tensor[test_indices], trajectories_tensor[test_indices], params_tensor[test_indices], fitted_trajectories_tensor[test_indices], img_transform=transform_test)

    # Create DataLoader for each subset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Dataset size: {dataset_size}")
    print(f"Train size: {train_size}, Validation size: {val_size}, Test size: {test_size}")

    return train_loader, val_loader, test_loader

def visualize_samples(sketches1, sketches2, original_points, fitted_trajectory, img_name, dense_fitted_trajectory=None):
    num_samples = len(sketches1)
    # fig, axes = plt.subplots(5, num_samples, figsize=(3*num_samples, 6))
    subplot_num = 3 if dense_fitted_trajectory is None else 5
    fig = plt.figure(figsize=(subplot_num*num_samples, 10))

    # Create a colormap
    cmap = plt.get_cmap('viridis')

    for i in range(num_samples):
        ax1 = fig.add_subplot(subplot_num, num_samples, i+1)
        ax1.imshow(np.flipud(sketches1[i].squeeze()))
        ax1.set_title(f"Sample {i+1}: Sketch 1")
        ax1.set_aspect('equal', 'box')

        ax2 = fig.add_subplot(subplot_num, num_samples, num_samples+i+1)
        ax2.imshow(np.flipud(sketches2[i].squeeze()))
        ax2.set_title(f"Sample {i+1}: Sketch 2")
        ax2.set_aspect('equal', 'box')
        
        ax3 = fig.add_subplot(subplot_num, num_samples, 2*num_samples+i+1, projection='3d')
        ax3.scatter(original_points[i][:, 0], original_points[i][:, 1], original_points[i][:, 2], c=np.arange(len(original_points[i])), cmap=cmap, alpha=0.3)
        ax3.plot(fitted_trajectory[i][:, 0], fitted_trajectory[i][:, 1], fitted_trajectory[i][:, 2], 'r-', linewidth=2)
        ax3.set_title(f"Sample {i+1}: Fitted Trajectory")

        if dense_fitted_trajectory is not None:
            ax4 = fig.add_subplot(subplot_num, num_samples, 3*num_samples+i+1, projection='3d')
            ax4.scatter(original_points[i][:, 0], original_points[i][:, 1], original_points[i][:, 2], c=np.arange(len(original_points[i])), cmap=cmap, alpha=0.3)
            ax4.plot(dense_fitted_trajectory[i][:, 0], dense_fitted_trajectory[i][:, 1], dense_fitted_trajectory[i][:, 2], 'r-', linewidth=2)
            ax4.set_title(f"Sample {i+1}: Dense Fitted Trajectory")

            ax5 = fig.add_subplot(subplot_num, num_samples, 4*num_samples+i+1, projection='3d')
            ax5.plot(fitted_trajectory[i][:, 0], fitted_trajectory[i][:, 1], fitted_trajectory[i][:, 2], 'r-', linewidth=2)
            ax5.plot(dense_fitted_trajectory[i][:, 0], dense_fitted_trajectory[i][:, 1], dense_fitted_trajectory[i][:, 2], 'g-', linewidth=2)
            ax5.set_title(f"Sample {i+1}: Comparison")
    
    plt.tight_layout()
    plt.savefig(f"{img_name}.png")


def load_samples_and_save(img_size=64, batch_size=32, num_control_points=20, num_samples=None):
    stime = time.time()
    sketches1, sketches2, trajectories = load_data(num_samples=num_samples, img_size=img_size)
    sketches1_tensor = torch.from_numpy(sketches1).permute(0, 3, 1, 2).float()
    sketches2_tensor = torch.from_numpy(sketches2).permute(0, 3, 1, 2).float()
    print(f"Dataset generated with {len(sketches1_tensor)} samples.")
    print(f"Sketch shape: {sketches1_tensor.shape}")
    print(f"Sketch min: {sketches1_tensor.min()}", f"Sketch max: {sketches1_tensor.max()}")
    trajectories_tensor = torch.from_numpy(trajectories).float()

    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/'
    torch.save(sketches1_tensor, root_path + f'sketches1_{img_size}.pt')
    torch.save(sketches2_tensor, root_path + f'sketches2_{img_size}.pt')
    torch.save(trajectories_tensor, root_path + 'trajectories.pt')

    print(f"Saved data to {root_path}")
    print(f"Time taken: {time.time() - stime:.2f} sec")
    # Time taken: 302.55 sec


def fit_and_save_trajectories(num_control_points=20, model_type='bspline'):
    stime = time.time()
    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/'
    trajectories = torch.load(root_path + 'trajectories.pt')
    # breakpoint()

    params, fitted_trajectories = get_trajectory_params_bspline(trajectories, num_control_points=num_control_points)
    params_tensor = torch.from_numpy(params).float()
    fitted_trajectories_tensor = torch.from_numpy(fitted_trajectories).float()
    
    print(f"Trajectory shape: {trajectories.shape}")
    print(f"Params shape: {params_tensor.shape}")
    print(f"Fitted Trajectory shape: {fitted_trajectories.shape}")
    
    torch.save(params_tensor, root_path + f'params_{num_control_points}.pt')
    torch.save(fitted_trajectories_tensor, root_path + f'fitted_trajectories_{num_control_points}.pt')

    print(f"Saved data to {root_path}")
    print(f"Time taken: {time.time() - stime:.2f} sec")
    # Time taken: 8.58 sec


def sample_and_visualize(model_type='bspline'):
    sketches1, sketches2, trajectories = load_data(num_samples=6, img_size=64)
    if model_type == 'bspline':
        params, fitted_trajectories = get_trajectory_params_bspline(trajectories, num_control_points=20)
        dense_params, dense_fitted_trajectories = get_trajectory_params_bspline(trajectories, num_control_points=50)
        # dense ones fit better
    else:
        params, fitted_trajectories = get_trajectory_params_seb(trajectories, num_basis=20)
    visualize_samples(sketches1, sketches2, trajectories, fitted_trajectories, "sample", dense_fitted_trajectory=dense_fitted_trajectories)


if __name__ == "__main__":
    # sample_and_visualize(model_type="bspline")
    # sample_and_visualize(model_type="seb")
    # load_samples_and_save()
    fit_and_save_trajectories(num_control_points=50)