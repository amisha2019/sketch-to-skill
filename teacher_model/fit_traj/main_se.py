import h5py
import numpy as np
import matplotlib.pyplot as plt
from fit_traj.fit_traj_seb import fit_trajectory, trajectory_model
from bspline_noise_scaled import get_noisy_curve


def visualize_results(original_points, fitted_trajectory, noisy_trajectories, img_name):
    fig = plt.figure(figsize=(20, 5))
    
    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    # Original points
    ax1 = fig.add_subplot(141, projection='3d')
    scatter1 = ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                           c=np.arange(len(original_points)), cmap=cmap)
    ax1.set_title('Original Trajectory')
    fig.colorbar(scatter1, ax=ax1, label='Point Order')
    
    # Fitted trajectory
    ax2 = fig.add_subplot(142, projection='3d')
    scatter2 = ax2.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                           c=np.arange(len(original_points)), cmap=cmap, alpha=0.3)
    ax2.plot(fitted_trajectory[:, 0], fitted_trajectory[:, 1], fitted_trajectory[:, 2], 'r-*', linewidth=2)
    ax2.set_title('Fitted Trajectory')
    fig.colorbar(scatter2, ax=ax2, label='Point Order')

    # Noisy trajectories
    for i, noisy_trajectory in enumerate(noisy_trajectories):
        ax = fig.add_subplot(143 + i, projection='3d')
        scatter = ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                           c=np.arange(len(original_points)), cmap=cmap, alpha=0.3)
        ax.plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], noisy_trajectory[:, 2], 'r-*', linewidth=2)
        ax.set_title(f'Noisy Trajectory {i+1}')
        fig.colorbar(scatter, ax=ax, label='Point Order')
    
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{img_name}.png")

def main(original_points, img_name):
    
    num_basis = 10
    fitted_weights = fit_trajectory(points, num_basis)
    print(f"Range of fitted weights: {np.min(fitted_weights)} to {np.max(fitted_weights)}")
    
    # Generate smooth trajectory for plotting
    s_smooth = np.linspace(0, 1, 200)
    # breakpoint()
    fitted_trajectory = trajectory_model(s_smooth, fitted_weights, num_basis)

    noise_level = 0.01
    noisy_trajectories = []
    for i in range(2):
        # Generate noisy trajectory for plotting
        noisy_weights = fitted_weights + np.random.normal(0, noise_level, fitted_weights.shape)
        noisy_trajectory = trajectory_model(s_smooth, noisy_weights, num_basis)
        noisy_trajectories.append(noisy_trajectory)
    
    # Visualize results
    visualize_results(original_points, fitted_trajectory, noisy_trajectories, img_name)

if __name__ == "__main__":

    file = "./data/metaworld/Assembly_frame_stack_1_96x96_end_on_success/dataset.hdf5"
    data = h5py.File(file, 'r')
    print(list(data.keys()))
    group = data['data']
    print(list(group.keys()))

    all_points = []
    for key in group.keys():
        demo = group[key]
        # print(list(demo.keys()))
        # print(demo["states"].shape)
        points = demo["states"][:, :3]
        all_points.append(points)
    
        main(points, key)