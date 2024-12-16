# This code implements an algorithm for fitting a 3D trajectory. Below is a brief description of the code and the algorithm:

# Code Description:
# The program uses a trajectory model based on squared exponential basis functions to fit 3D point trajecotry data. It first generates a complex spiral trajectory as test data, then applies spectral clustering to sort the points. Next, it uses least squares optimization to fit the trajectory model, and finally visualizes the original points and the fitted trajectory.

# Algorithm Description:
# 1. Data Generation and Preprocessing:
#    - Generate complex 3D spiral trajectory points.
#    - Use a spectral ordering method to sort the points and recover their sequence.
#    - Add random noise to simulate real data.

# 2. Trajectory Model:
#    - Use squared exponential basis functions as the basic building blocks.
#    - Represent the trajectory as a linear combination of the basis functions.

# 3. Fitting Process:
#    - Define the objective function as the sum of squared errors between the model-predicted points and the actual points.
#    - Use the L-BFGS-B optimization algorithm to minimize the objective function and obtain the optimal basis function weights.

# 4. Visualization:
#    - Plot the original points and the fitted trajectory in 3D space.
#    - Display the trajectory components along the X, Y, and Z dimensions.

# ### Advantages of this Approach:
# This method is advantageous in handling complex 3D trajectories and can effectively manage unordered point trajecotry data through the spectral clustering preprocessing step. Additionally, the model based on squared exponential basis functions offers good smoothness and flexibility.

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial import distance_matrix
from mpl_toolkits.mplot3d import Axes3D

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

def objective_function(weights, points, num_basis):
    """Compute the error between the model and the actual points."""
    weights = weights.reshape(-1, 3)
    s = np.linspace(0, 1, len(points))
    model_points = trajectory_model(s, weights, num_basis)
    return np.sum((model_points - points)**2)

def fit_trajectory(points, num_basis=10):
    """Fit a trajectory model to the given points."""
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

def spectral_ordering(points, k=10):
    # Construct k-nearest neighbor graph
    dist_matrix = distance_matrix(points, points)
    k_nearest = np.argsort(dist_matrix, axis=1)[:, 1:k+1]
    
    # Construct adjacency matrix
    n = len(points)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, k_nearest[i]] = 1
        A[k_nearest[i], i] = 1
    
    # Compute Laplacian
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    
    # Compute eigenvectors and eigenvalues
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Use the Fiedler vector (second smallest eigenvalue) for ordering
    fiedler_vector = eigenvectors[:, 1]
    return np.argsort(fiedler_vector)


if __name__ == "__main__":
    # # Generate some example 3D points (replace this with your actual data)
    # t = np.linspace(0, 2*np.pi, 100)
    # x = np.sin(t)
    # y = np.cos(t)
    # z = t

    # generate complex trajectory
    t = np.linspace(0, 4*np.pi, 200)
    x = t * np.cos(t)
    y = t * np.sin(t)
    z = t

    points = np.column_stack([x, y, z])
    np.random.shuffle(points)

    points = points[spectral_ordering(points)]

    # Add some noise to simulate real-world data
    points += np.random.normal(0, 0.05, points.shape)

    # Fit the trajectory
    fitted_weights = fit_trajectory(points)

    # Generate smooth trajectory for plotting
    s_smooth = np.linspace(0, 1, 200)
    fitted_trajectory = trajectory_model(s_smooth, fitted_weights)

    # Plot results
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c='b', label='Original Points')
    ax1.plot(fitted_trajectory[:, 0], fitted_trajectory[:, 1], fitted_trajectory[:, 2], 'r', label='Fitted Trajectory')
    ax1.set_title('3D View')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.plot(s_smooth, fitted_trajectory)
    ax2.set_title('Trajectory Components')
    ax2.legend(['X', 'Y', 'Z'])

    plt.tight_layout()
    plt.show()