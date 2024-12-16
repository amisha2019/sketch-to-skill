import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev


def fit_and_resample_trajectory(trajectory, num_control_points=20, smoothness=0.05):
    k = 3
    n_knots = num_control_points + k + 1
    knots = np.clip(np.linspace(0, 1, n_knots), 0, 1)
    knots[:k+1] = 0
    knots[-k-1:] = 1
    # Fit a smoothing spline to the trajectory
    tck, u = splprep(trajectory.T, s=smoothness, k=k, t=knots)
    # Resample the trajectory with the desired number of control points
    u_new = np.linspace(0, 1, len(trajectory))
    new_trajectory = splev(u_new, tck)
    # Keep end points unchanged
    new_trajectory = np.array(new_trajectory).T
    # new_trajectory[0] = trajectory[0]
    # new_trajectory[-1] = trajectory[-1]
    return new_trajectory

def add_gaussian_noise(trajectory, scale=0.015):
    num_points = trajectory.shape[0]
    noise_scale = np.sin(np.linspace(0, np.pi, num_points))
    noise = np.random.normal(0, scale, size=trajectory.shape) * noise_scale[:, np.newaxis]
    noise[0] = 0 
    noise[-1] = 0  
    return trajectory + noise

def jitter_trajectory(trajectory, jitter_range=0.005):
    jittered = trajectory + np.random.uniform(-jitter_range, jitter_range, size=trajectory.shape)
    jittered[0] = trajectory[0]  # Keep the start point unchanged
    jittered[-1] = trajectory[-1]  # Keep the end point unchanged
    return jittered

def plot_trajectories(trajectories):
    rows = len(trajectories.keys())
    cols = len(trajectories[list(trajectories.keys())[0]])
    fig = plt.figure(figsize=(4*cols, 4*rows))
    cmap = plt.get_cmap('viridis')
    for i, (traj_type, traj_list) in enumerate(trajectories.items()):
        for j, traj in enumerate(traj_list):
            ax = fig.add_subplot(rows, cols, i*cols + j + 1, projection='3d')
            if isinstance(traj, list):
                for t in traj:
                    ax.plot(t[:, 0], t[:, 1], t[:, 2])
                # ax.scatter(t[:, 0], t[:, 1], t[:, 2], c=np.arange(len(t[:, 0])), cmap=cmap, alpha=0.3, label=traj_type.capitalize())
            else:
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2])
                ax.scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=np.arange(len(traj[:, 0])), cmap=cmap, alpha=0.3, label=traj_type.capitalize())
            ax.legend()
            ax.set_title(f'{traj_type.capitalize()} Trajectory {j+1}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('trajectory_augmentations.png')
    plt.close()

def test_trajectory_augment(img_size, num_control_points):
    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/'
    
    trajectories_tensor = torch.load(root_path + 'trajectories.pt')
    params_tensor = torch.load(root_path + f'params_{num_control_points}.pt')
    fitted_trajectories_tensor = torch.load(root_path + f'fitted_trajectories_50.pt')

    # Pick a random subset of examples
    num_examples = 5
    indices = np.random.choice(len(trajectories_tensor), num_examples, replace=False)
    
    all_trajectories = {
        'original': [],
        'fitted': [],
        'noisy': [],
        # 'jittered': []
    }

    for i, idx in enumerate(indices):
        original_trajectory = trajectories_tensor[idx].numpy()
        original_fitted = fitted_trajectories_tensor[idx].numpy()

        # Apply augmentations
        noisy_trajectories = []
        jittered_trajectories = []
        for j in range(6):
            noisy_trajectory = fit_and_resample_trajectory(add_gaussian_noise(original_trajectory))
            noisy_trajectories.append(noisy_trajectory)
            # jittered_trajectory = fit_and_resample_trajectory(jitter_trajectory(original_trajectory))
            # jittered_trajectories.append(jittered_trajectory)

        all_trajectories['original'].append(original_trajectory)
        all_trajectories['fitted'].append(original_fitted)
        all_trajectories['noisy'].append(noisy_trajectories)
        # all_trajectories['jittered'].append(jittered_trajectories)

    # Plotting the results
    plot_trajectories(all_trajectories)
    print(f"Tested {num_examples} random examples with trajectory augmentations.")

if __name__ == '__main__':
    test_trajectory_augment(img_size=64, num_control_points=20)