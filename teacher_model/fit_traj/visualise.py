import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fit_traj_bspline import fit_trajectory_spline, predict_trajectory_spline
from data import fit_trajectory_bspline, predict_trajectory_bspline

def visualize_results(original_points, fitted_trajectory, executed_trajectory, img_name):
    plt.clf()  # Clear the figure to ensure a fresh start for each plot
    fig = plt.figure(figsize=(18, 8))  # Adjust size to fit three subplots

    # Create a colormap
    cmap = plt.get_cmap('viridis')

    # Original points
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                c=np.arange(len(original_points)), cmap=cmap)
    ax1.set_title('Original Trajectory')

    # Fitted trajectory
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')
    ax2.plot(fitted_trajectory[:, 0], fitted_trajectory[:, 1], fitted_trajectory[:, 2], 'r-*', linewidth=2)
    ax2.set_title('Fitted Trajectory')

    # Executed trajectory
    ax3 = fig.add_subplot(1, 3, 3, projection='3d')
    ax3.plot(executed_trajectory[:, 0], executed_trajectory[:, 1], executed_trajectory[:, 2], 'b-*', linewidth=2)
    ax3.set_title('Executed Trajectory')
    
    plt.tight_layout()
    plt.savefig(f"{img_name}.png")

def main(original_points, img_name):
    num_control_points = 10
    tck, u = fit_trajectory_bspline(original_points, num_control_points)
    num_points = len(original_points)
    fitted_trajectory = predict_trajectory_bspline(tck, num_points=num_points)

    # Load executed trajectory from the additional HDF5 file
    executed_file = "release/data/metaworld/ButtonPresss_frame_stack_1_96x96_end_on_success/trajectories_rand.hdf5"
    executed_data = h5py.File(executed_file, 'r')
    executed_trajectory = executed_data['data'][img_name]['obs']['prop'][:]
    # print("executed_trajectory", executed_data['data'][img_name]['obs']['prop'][:])
    executed_data.close()

    visualize_results(original_points, fitted_trajectory, executed_trajectory, img_name)

if __name__ == "__main__":
    file = "release/data/metaworld/ButtonPresss_frame_stack_1_96x96_end_on_success/dataset.hdf5"
    data = h5py.File(file, 'r')
    group = data['data']
    for key in group.keys():
        demo = group[key]
        points = demo["states"][:, :3]
        img_name = key  # Use key as image name
        main(points, img_name)

    data.close()
