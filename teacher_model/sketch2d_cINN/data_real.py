import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy import interpolate
from scipy.interpolate import splprep, splev
import h5py

def spectral_ordering(points, k=10):
    from scipy.spatial import distance_matrix
    
    dist_matrix = distance_matrix(points, points)
    k_nearest = np.argsort(dist_matrix, axis=1)[:, 1:k+1]
    
    n = len(points)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, k_nearest[i]] = 1
        A[k_nearest[i], i] = 1
    
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    fiedler_vector = eigenvectors[:, 1]
    return np.argsort(fiedler_vector)


def fit_trajectory_spline(points, num_control_points=20, s=0, k=3):
    u = np.linspace(0, 1, len(points))
    n_knots = num_control_points + k + 1
    knots = np.clip(np.linspace(0, 1, n_knots), 0, 1)
    knots[:k+1] = 0
    knots[-k-1:] = 1
    if points.shape[1] == 2:
        tck, u = splprep([points[:, 0], points[:, 1]], u=u, s=s, k=k, t=knots, task=-1)
    else:
        tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], u=u, s=s, k=k, t=knots, task=-1)
    # print(knots)
    # print(tck[0])
    return tck, u

def predict_trajectory_spline(tck, num_points=200):
    u_new = np.linspace(0, 1, num_points)
    return np.array(splev(u_new, tck)).T


def load_data(file_path=None, num_samples=None):
    if file_path is None:
        file_path = '/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_datasets/sketch_data_assembly.hdf5'
    corner_images = []

    # Load the dataset
    with h5py.File(file_path, 'r') as f:
        print(f"Loading data from {file_path}")
        length = len(list(f.keys()))
        if num_samples is not None:
            length = num_samples

        for i in range(length):
            img = f[f'demo_{i}/corner'][:]
            corner_images.append(img)

    corner_images = np.array(corner_images)
    print(f"Loaded {len(corner_images)} images")
    print(f"Image shape: {corner_images.shape}")
    return corner_images


def get_trajectory_params(images, num_control_points=20):
    trajectories = []
    params = []
    fitted_trajectories = []

    for img in images:
        non_zero_indices = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if np.any(img[i, j] != 0):
                    non_zero_indices.append([i, j])
        points = np.array(non_zero_indices)

        ordered_indices = spectral_ordering(points)
        ordered_points = points[ordered_indices]

        tck, u = fit_trajectory_spline(ordered_points, num_control_points)
        fitted_trajectory = predict_trajectory_spline(tck, num_points=200)

        trajectories.append(ordered_points)
        params.append(np.array(tck[1]).flatten())
        fitted_trajectories.append(fitted_trajectory)

    return np.array(params), np.array(fitted_trajectories)


def get_dataloader(batch_size=32, num_control_points=20, num_samples=None):
    sketches = load_data(num_samples=num_samples)
    sketches_tensor = torch.from_numpy(sketches).permute(0, 3, 1, 2).float() / 255.0
    print(f"Dataset generated with {len(sketches)} samples.")
    print(f"Sketch shape: {sketches_tensor.shape}")
    print(f"Sketch min: {sketches_tensor.min()}", f"Sketch max: {sketches_tensor.max()}")

    params, fitted_trajectories = get_trajectory_params(sketches, num_control_points=num_control_points)
    # trajectories_tensor = torch.from_numpy(trajectories).float()
    params_tensor = torch.from_numpy(params).float()
    fitted_trajectories_tensor = torch.from_numpy(fitted_trajectories).float()
    # dataset = TensorDataset(sketches_tensor, trajectories_tensor)
    dataset = TensorDataset(sketches_tensor, params_tensor, fitted_trajectories_tensor)
    
    print(f"Params shape: {params_tensor.shape}")
    print(f"Fitted Trajectory shape: {fitted_trajectories.shape}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)