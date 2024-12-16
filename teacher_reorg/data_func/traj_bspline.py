import torch
import numpy as np
from scipy.interpolate import splprep, splev
from tqdm import tqdm


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

def get_bspline_basis_matrix(num_control_points, degree, num_points):
    n_knots = num_control_points + degree + 1
    knots = np.clip(np.linspace(0, 1, n_knots), 0, 1)
    knots[:degree+1] = 0
    knots[-degree-1:] = 1
    # get the basis matrix
    n = num_control_points - 1
    t_values = np.linspace(knots[degree], knots[n+1], num_points)
    basis_matrix = np.array([[bspline_basis(i, degree, t, knots) for i in range(n+1)] for t in t_values])
    basis_matrix[-1, -1] = 1.0  # ensure the last point is included
    return basis_matrix

def predict_trajectory_bspline_custom_2(basis_matrix, control_points):
    control_points = np.array(control_points).T
    points = np.dot(basis_matrix, control_points)
    return points

def predict_trajectory_bspline_custom(tck, num_points):
    knots, control_points, degree = tck
    control_points = np.array(control_points).T
    
    # get the basis matrix
    n = len(control_points) - 1
    t_values = np.linspace(knots[degree], knots[n+1], num_points)
    basis_matrix = np.array([[bspline_basis(i, degree, t, knots) for i in range(n+1)] for t in t_values])
    basis_matrix[-1, -1] = 1.0  # ensure the last point is included
    
    points = np.dot(basis_matrix, control_points)
    
    return points

def predict_trajectory_bspline_scipy(tck, num_points):
    u_new = np.linspace(0, 1, num_points)
    return np.array(splev(u_new, tck)).T

def fit_trajectory_bspline_scipy_orig(points, s=0, k=3):

    u = np.linspace(0, 1, len(points))
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    try:
        if points.shape[1] == 2:
            tck, u = splprep([points[:, 0], points[:, 1]], u=u, s=s, k=k)
        elif points.shape[1] == 3:
            tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], u=u, s=s, k=k)
        else:
            raise ValueError(f"Expected 2D or 3D points (n x 2 or n x 3), but got {points.shape}")
    except Exception as e:
        print(f"Error fitting B-spline: {e}")
        breakpoint()
    
    return tck, u

def fit_trajectory_bspline_scipy(points, num_control_points, s=0, k=3):
    if num_control_points > 20:
        # if points.shape[0] < num_control_points:
        #     breakpoint()
        num_control_points = min(num_control_points, points.shape[0] - k)

    u = np.linspace(0, 1, len(points))
    n_knots = num_control_points + k + 1
    knots = np.clip(np.linspace(0, 1, n_knots), 0, 1)
    knots[:k+1] = 0
    knots[-k-1:] = 1
    if isinstance(points, torch.Tensor):
        points = points.numpy()
    try:
        if points.shape[1] == 2:
            tck, u = splprep([points[:, 0], points[:, 1]], u=u, s=s, k=k, t=knots, task=-1)
        elif points.shape[1] == 3:
            tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], u=u, s=s, k=k, t=knots, task=-1)
        else:
            raise ValueError(f"Expected 2D or 3D points (n x 2 or n x 3), but got {points.shape}")
    except Exception as e:
        print(f"Error fitting B-spline: {e}")
        breakpoint()
    
    return tck, u


def interpolate_trajectory(trajectory, target_length):
    # trajectory: numpy array of shape (num_samples, 3)
    # target_length: int
    # return: numpy array of shape (target_length, 3)
    num_samples = len(trajectory)
    if num_samples == target_length:
        return trajectory
    
    # Create evenly spaced points for interpolation
    x = np.linspace(0, num_samples - 1, num_samples)
    x_new = np.linspace(0, num_samples - 1, target_length)
    
    # Perform linear interpolation for each dimension
    interpolated_trajectory = np.zeros((target_length, 3))
    for dim in range(3):
        interpolated_trajectory[:, dim] = np.interp(x_new, x, trajectory[:, dim])
    
    return interpolated_trajectory

def get_trajectory_params_bspline(trajectories, num_control_points, use_uniform_knots=True):
    params = []
    fitted_trajectories = []

    for i, points in tqdm(enumerate(trajectories), total=len(trajectories), desc="Fitting trajectories"):

        if points.shape[0] < num_control_points:
            points = interpolate_trajectory(points, num_control_points+5)

        if use_uniform_knots:
            tck, u = fit_trajectory_bspline_scipy(points, num_control_points)
        else:
            tck, u = fit_trajectory_bspline_scipy_orig(points)
        fitted_trajectory = predict_trajectory_bspline_scipy(tck, num_points=100)
        params.append(np.array(tck[1]).flatten())
        fitted_trajectories.append(fitted_trajectory)
    try:
        return np.array(params), np.array(fitted_trajectories)
    except:
        return None, np.array(fitted_trajectories)

def compare_bspline_methods(trajectories, num_control_points, num_points):
    params = []
    fitted_trajectories = []

    basis_matrix = get_bspline_basis_matrix(num_control_points, degree=3, num_points=num_points)
    for i, points in enumerate(trajectories):
        points = points.numpy()
        tck, u = fit_trajectory_bspline_scipy(points, num_control_points)
        
        fitted_trajectory_scipy = predict_trajectory_bspline_scipy(tck, num_points)
        # fitted_trajectory_custom = predict_trajectory_bspline_custom(tck, num_points)
        fitted_trajectory_custom = predict_trajectory_bspline_custom_2(basis_matrix, tck[1])

        if not np.allclose(fitted_trajectory_scipy, fitted_trajectory_custom):
            print(f"Trajectory {i} does not match.")
            print(np.mean(np.abs(fitted_trajectory_scipy - fitted_trajectory_custom)))
            breakpoint()
        else:
            print(f"Trajectory {i} matches.")
        
    return np.array(params), np.array(fitted_trajectories)


if __name__ == "__main__":
    trajectories = torch.load('/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/trajectories.pt')
    compare_bspline_methods(trajectories, num_control_points=20, num_points=100)
