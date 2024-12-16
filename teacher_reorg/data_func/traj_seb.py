import numpy as np
from scipy.optimize import minimize
from data import get_chamfer_distance


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

