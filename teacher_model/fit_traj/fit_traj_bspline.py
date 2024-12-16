import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors

def generate_complex_trajectory(num_points=200, noise_level=0.05):
    t = np.linspace(0, 4*np.pi, num_points)
    x = t * np.cos(t) + np.random.normal(0, noise_level, num_points)
    y = t * np.sin(t) + np.random.normal(0, noise_level, num_points)
    z = t + np.sin(3*t) + np.random.normal(0, noise_level, num_points)
    return np.column_stack([x, y, z])

def shuffle_points(points):
    indices = np.arange(len(points))
    np.random.shuffle(indices)
    return points[indices], indices

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

# def fit_trajectory_spline(points, s=0, k=3):
#     t = np.linspace(0, 1, len(points))
#     tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], u=t, s=s, k=k)
#     return tck, u

def fit_trajectory_spline(points, num_control_points=20, s=0, k=3):
    u = np.linspace(0, 1, len(points))
    n_knots = num_control_points + k + 1
    knots = np.clip(np.linspace(0, 1, n_knots), 0, 1)
    knots[:k+1] = 0
    knots[-k-1:] = 1
    tck, u = splprep([points[:, 0], points[:, 1], points[:, 2]], u=u, s=s, k=k, t=knots, task=-1)
    # print(knots)
    # print(tck[0])
    return tck, u

def predict_trajectory_spline(tck, num_points=200):
    u_new = np.linspace(0, 1, num_points)
    return np.array(splev(u_new, tck)).T

def arc_length(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    return np.sqrt(dx**2 + dy**2)

def downsample_tck(tck, num_control_points):
    u_dense = np.linspace(0, 1, 400)
    x_dense, y_dense = splev(u_dense, tck)

    distances = arc_length(x_dense, y_dense)
    cumulative_arc_length = np.cumsum(distances)
    total_arc_length = cumulative_arc_length[-1]
    normalized_arc_length = cumulative_arc_length / total_arc_length  # 归一化弧长

    # 修复：对齐 u_dense 的长度，添加首点以匹配 u_dense 长度
    normalized_arc_length = np.insert(normalized_arc_length, 0, 0)


def visualize_results(original_points, shuffled_points, shuffle_indices, ordered_points, fitted_trajectory):
    fig = plt.figure(figsize=(20, 5))
    
    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    # Original points
    ax1 = fig.add_subplot(141, projection='3d')
    scatter1 = ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                           c=np.arange(len(original_points)), cmap=cmap)
    ax1.set_title('Original Trajectory')
    fig.colorbar(scatter1, ax=ax1, label='Point Order')
    
    # Shuffled points
    ax2 = fig.add_subplot(142, projection='3d')
    scatter2 = ax2.scatter(shuffled_points[:, 0], shuffled_points[:, 1], shuffled_points[:, 2],
                           c=np.arange(len(ordered_points)), cmap=cmap)
    ax2.set_title('Shuffled Points')
    fig.colorbar(scatter2, ax=ax2, label='Shuffle Index')
    
    # Ordered points
    ax3 = fig.add_subplot(143, projection='3d')
    scatter3 = ax3.scatter(ordered_points[:, 0], ordered_points[:, 1], ordered_points[:, 2],
                           c=np.arange(len(ordered_points)), cmap=cmap)
    ax3.set_title('Ordered Points')
    fig.colorbar(scatter3, ax=ax3, label='Recovered Order')
    
    # Fitted trajectory
    ax4 = fig.add_subplot(144, projection='3d')
    scatter4 = ax4.scatter(ordered_points[:, 0], ordered_points[:, 1], ordered_points[:, 2],
                           c=np.arange(len(ordered_points)), cmap=cmap, alpha=0.3)
    ax4.plot(fitted_trajectory[:, 0], fitted_trajectory[:, 1], fitted_trajectory[:, 2], 'r-*', linewidth=2)
    ax4.set_title('Fitted Trajectory')
    fig.colorbar(scatter4, ax=ax4, label='Point Order')
    
    plt.tight_layout()
    plt.show()

def visualize_results_2(original_points, shuffled_points, shuffle_indices, ordered_points, fitted_trajectory):
    fig = plt.figure(figsize=(20, 5))
    
    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    # Original points
    ax1 = fig.add_subplot(141, projection='3d')
    scatter1 = ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                           c=np.arange(len(original_points)), cmap=cmap)
    ax1.set_title('Original Trajectory')
    fig.colorbar(scatter1, ax=ax1, label='Point Order')
    
    # Shuffled points
    ax2 = fig.add_subplot(142, projection='3d')
    scatter2 = ax2.scatter(shuffled_points[:, 0], shuffled_points[:, 1], shuffled_points[:, 2],
                           c=np.arange(len(ordered_points)), cmap=cmap)
    ax2.set_title('Shuffled Points')
    fig.colorbar(scatter2, ax=ax2, label='Shuffle Index')
    
    # Ordered points
    ax3 = fig.add_subplot(143, projection='3d')
    scatter3 = ax3.scatter(ordered_points[:, 0], ordered_points[:, 1], ordered_points[:, 2],
                           c=np.arange(len(ordered_points)), cmap=cmap)
    ax3.set_title('Ordered Points')
    fig.colorbar(scatter3, ax=ax3, label='Recovered Order')
    
    # Fitted trajectory
    ax4 = fig.add_subplot(144, projection='3d')
    scatter4 = ax4.scatter(ordered_points[:, 0], ordered_points[:, 1], ordered_points[:, 2],
                           c=np.arange(len(ordered_points)), cmap=cmap, alpha=0.3)
    ax4.plot(fitted_trajectory[:, 0], fitted_trajectory[:, 1], fitted_trajectory[:, 2], 'r-*', linewidth=2)
    ax4.set_title('Fitted Trajectory')
    fig.colorbar(scatter4, ax=ax4, label='Point Order')
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate complex trajectory
    original_points = generate_complex_trajectory()
    import fit_traj_test as fit_traj_test
    original_points = fit_traj_test.all_points[1]
    
    # Shuffle points
    shuffled_points, shuffle_indices = shuffle_points(original_points)
    
    # Order points
    ordered_indices = spectral_ordering(shuffled_points)
    ordered_points = shuffled_points[ordered_indices]
    
    # Fit trajectory using spline
    tck, u = fit_trajectory_spline(original_points)
    
    # Generate smooth trajectory for plotting
    fitted_trajectory = predict_trajectory_spline(tck, num_points=20)
    
    # Visualize results
    visualize_results(original_points, shuffled_points, shuffle_indices, ordered_points, fitted_trajectory)

if __name__ == "__main__":
    main()