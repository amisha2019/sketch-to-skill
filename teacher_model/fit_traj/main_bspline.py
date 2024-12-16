import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fit_traj_bspline import fit_trajectory_spline, predict_trajectory_spline, visualize_results
from bspline_noise_scaled import get_noisy_curve
from env.metaworld_wrapper import PixelMetaWorld
def visualize_results(original_points, fitted_trajectory, noisy_trajectories, img_name):
    fig = plt.figure(figsize=(25, 10))  # Adjust size to better fit more subplots

    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    # Original points
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')  # Row 1, Col 1
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                c=np.arange(len(original_points)), cmap=cmap)
    ax1.set_title('Original Trajectory')

    # Fitted trajectory
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')  # Row 1, Col 2
    ax2.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                c=np.arange(len(original_points)), cmap=cmap, alpha=0.3)
    ax2.plot(fitted_trajectory[:, 0], fitted_trajectory[:, 1], fitted_trajectory[:, 2], 'r-*', linewidth=2)
    ax2.set_title('Fitted Trajectory')

    # Noisy trajectories
    for i, noisy_trajectory in enumerate(noisy_trajectories):
        ax = fig.add_subplot(2, 4, 3 + i, projection='3d')  # Dynamic positioning in Row 1, Col 3+
        ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                   c=np.arange(len(original_points)), cmap=cmap, alpha=0.3)
        ax.plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], noisy_trajectory[:, 2], 'r-*', linewidth=2)
        ax.set_title(f'Noisy Trajectory {i+1}')
    
    plt.tight_layout()
    plt.savefig(f"{img_name}.png")


def compute_actions(trajectory):
    """Compute delta positions as actions from a given trajectory."""
    actions = np.diff(trajectory, axis=0)
    return np.clip(actions, -1, 1) 

def save_trajectories_to_csv(original_points, fitted_trajectory, noisy_trajectories, prefix):
    # Save original points
    pd.DataFrame(original_points, columns=['x', 'y', 'z']).to_csv(f"{prefix}_original.csv", index=False)

    # Save fitted trajectory
    pd.DataFrame(fitted_trajectory, columns=['x', 'y', 'z']).to_csv(f"{prefix}_fitted.csv", index=False)

    # Save noisy trajectories
    for i, trajectory in enumerate(noisy_trajectories):
        pd.DataFrame(trajectory, columns=['x', 'y', 'z']).to_csv(f"{prefix}_noisy_{i+1}.csv", index=False)


def save_hdf5(original_points, fitted_trajectory, noisy_trajectories, prefix):
    actions_fitted = compute_actions(fitted_trajectory)
    actions_original = compute_actions(original_points)
    actions_noisy = [compute_actions(noisy) for noisy in noisy_trajectories]
    with h5py.File("trajectories.hdf5", 'a') as f:
        # Create or get the 'data' group
        data_group = f.require_group('data')

        def create_or_replace(group, name, data):
            full_path = f'{group.name}/{name}'
            if full_path in f:
                del f[full_path]
            group.create_dataset(name, data=data)

        # Use the 'data_group' instead of 'f' to create datasets within the 'data' group
        # create_or_replace(data_group, f'{prefix}_original/obs/prop', original_points)
        #dones = np.zeros(len(original_points))
        dones = np.zeros(len(original_points) - 1)
        dones[-1] = 1
        create_or_replace(data_group, f'{prefix}/actions', actions_original)
        create_or_replace(data_group, f'{prefix}/rewards', dones)
        create_or_replace(data_group, f'{prefix}/dones', dones)
        # create_or_replace(data_group, f'{prefix}/actions', actions_fitted)      # Consider only fitted trajectories
        # create_or_replace(data_group, f'{prefix}/rewards', dones)
        # create_or_replace(data_group, f'{prefix}/dones', dones)
        # for i, trajectory in enumerate(noisy_trajectories):
        #     create_or_replace(data_group, f'{prefix}_noisy_{i+1}/actions', actions)
        #     create_or_replace(data_group, f'{prefix}_noisy_{i+1}/rewards', dones)
        #     create_or_replace(data_group, f'{prefix}_noisy_{i+1}/dones', dones)
        print(f"Saved {prefix} to HDF5 file")


            

def get_images_for_hdf5(hdf5_file):
    env_params = dict(
        env_name="CoffeePush",
        robots=["Sawyer"],
        episode_length=100,
        action_repeat=2,
        frame_stack=1,
        obs_stack=1,
        reward_shaping=False,
        rl_image_size=96,
        camera_names=["corner2", "corner"],
        rl_camera="corner2",
        device="cuda",
        use_state=True,
    )
    env = PixelMetaWorld(**env_params)
    with h5py.File(hdf5_file, 'a') as f:
        data_group = f.require_group('data')  # Ensure the 'data' group exists
        keys = list(data_group.keys())
        for key in keys:
            if f'{key}/actions' in data_group:
                actions = np.array(data_group[f'{key}/actions'])
                image_group_key = f'{key}/obs/corner2_image'
                prop_key = f'{key}/obs/prop'

                props = []

                rl_obs, image_obs = env.reset()
                initial_img = image_obs['corner2'].cpu().numpy()

                if image_group_key in data_group:
                    img_dataset = data_group[image_group_key]
                else:
                    max_image_shape = (None,) + initial_img.shape
                    img_dataset = data_group.create_dataset(image_group_key, shape=(0,) + initial_img.shape, maxshape=max_image_shape, dtype='uint8')

                img_index = 0
                for action in actions:
                    rl_obs, reward, terminal, _, image_obs = env.step(action)
                    image = image_obs['corner2'].cpu().numpy()
                    props.append(rl_obs["prop"].cpu().numpy()[:3])     # Store the executed prop
                    img_dataset.resize((img_dataset.shape[0] + 1,) + image.shape)
                    img_dataset[img_index] = image
                    img_index += 1
                    if terminal:
                        break
                print(f"Executed props: {prop_key}")
                # if exists delete and create again
                if prop_key in data_group:
                    del data_group[prop_key]
                data_group.create_dataset(prop_key, data=props)
                

                if img_dataset.shape[0] > len(actions):
                    img_dataset.resize((len(actions),) + initial_img.shape)







def main(original_points, img_name):
    num_control_points = 10
    tck, u = fit_trajectory_spline(original_points, num_control_points)
    num_points = len(original_points)
    fitted_trajectory = predict_trajectory_spline(tck, num_points=num_points)
    noise_level = 0.01
    noisy_trajectories = []
    for i in range(5):
        noisy_trajectory = get_noisy_curve(tck, noise_level, num_points=num_points)
        noisy_trajectories.append(noisy_trajectory)
    visualize_results(original_points, fitted_trajectory, noisy_trajectories, img_name)
    save_hdf5(original_points, fitted_trajectory, noisy_trajectories, img_name)

if __name__ == "__main__":
    file = "release/data/metaworld/ButtonPresss_frame_stack_1_96x96_end_on_success/dataset.hdf5"
    data = h5py.File(file, 'r')
    group = data['data']
    print(list(group.keys()))
    all_points = []
    for key in group.keys():
        demo = group[key]
        points = demo["states"][:, :3]
        all_points.append(points)
        # main(points, image_name, key)
        main(points, key)
    get_images_for_hdf5("trajectories.hdf5")  # Calling the function to capture and store images


data.close()  