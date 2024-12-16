import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fit_traj_bspline import fit_trajectory_spline, predict_trajectory_spline, visualize_results
from bspline_noise_scaled import get_noisy_curve
from env.metaworld_wrapper import PixelMetaWorld
import time
def visualize_results(original_points, fitted_trajectory, noisy_trajectories, img_name):
    fig = plt.figure(figsize=(25, 10))  # Adjust size to better fit more subplots

    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    # Original points
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')  # Row 1, Col 1
    ax1.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                c=np.arange(len(original_points)), cmap=cmap)
    ax1.set_title('Original Trajectory')

    # Fitted trajectory
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')  # Row 1, Col 2
    ax2.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
                c=np.arange(len(original_points)), cmap=cmap, alpha=0.3)
    ax2.plot(fitted_trajectory[:, 0], fitted_trajectory[:, 1], fitted_trajectory[:, 2], 'r-*', linewidth=2)
    ax2.set_title('Fitted Trajectory')

    # Noisy trajectories
    # for i, noisy_trajectory in enumerate(noisy_trajectories):
    #     ax = fig.add_subplot(2, 4, 3 + i, projection='3d')  # Dynamic positioning in Row 1, Col 3+
    #     ax.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2],
    #                c=np.arange(len(original_points)), cmap=cmap, alpha=0.3)
    #     # ax.plot(noisy_trajectory[:, 0], noisy_trajectory[:, 1], noisy_trajectory[:, 2], 'r-*', linewidth=2)
    #     ax.set_title(f'Noisy Trajectory {i+1}')
    
    plt.tight_layout()
    plt.savefig(f"{img_name}.png")



def save_trajectories_to_csv(original_points, fitted_trajectory, noisy_trajectories, prefix):
    # Save original points
    pd.DataFrame(original_points, columns=['x', 'y', 'z']).to_csv(f"{prefix}_original.csv", index=False)

    # Save fitted trajectory
    pd.DataFrame(fitted_trajectory, columns=['x', 'y', 'z']).to_csv(f"{prefix}_fitted.csv", index=False)

    # Save noisy trajectories
    for i, trajectory in enumerate(noisy_trajectories):
        pd.DataFrame(trajectory, columns=['x', 'y', 'z']).to_csv(f"{prefix}_noisy_{i+1}.csv", index=False)


def save_hdf5(original_points, fitted_trajectory, noisy_trajectories, prefix, obj_pose, last_reached):
    with h5py.File("trajectories.hdf5", 'a') as f:
        data_group = f.require_group('data')

        def create_or_replace(group, name, data):
            full_path = f'{group.name}/{name}'
            if full_path in f:
                del f[full_path]
            group.create_dataset(name, data=data)

        dones = np.zeros(len(original_points))
        dones[-1] = 1

        create_or_replace(data_group, f'{prefix}/dones', dones)
        
        print(f"Saving {prefix} to HDF5 file")

        env_params = {
            "env_name": env_name,
            "robots": ["Sawyer"],
            "episode_length": 100,
            "action_repeat": 2,
            "frame_stack": 1,
            "obs_stack": 1,
            "reward_shaping": False,
            "rl_image_size": 96,
            "camera_names": ["corner2", "corner"],
            "rl_camera": "corner2",
            "device": "cuda",
            "use_state": True,
        }
        env = PixelMetaWorld(**env_params)
        rl_obs, image_obs = env.reset()

        # if env_name == "Reach":
        #     env.obj_rand_init(False)
        #     env.env.env.env.mw_set_goal_pose(obj_pose)
        #     env.env.env.env.mw_reset_model()
        # else:
        #     env.obj_rand_init(False)
        #     env.set_obj_pose_sktchRL(obj_pose)

        prop = fitted_trajectory
        image_group_key = f'{prefix}/obs/corner2_image'
        action_key = f'{prefix}/actions'
        reward_key = f'{prefix}/rewards'
        prop_key = f'{prefix}/obs/prop'

        actions = []
        rewards = []
        new_prop = []
        states = []


        rl_obs, image_obs = env.reset()
        initial_img = image_obs['corner2'].cpu().numpy()

        if image_group_key in data_group:
            img_dataset = data_group[image_group_key]
        else:
            max_image_shape = (None,) + initial_img.shape
            img_dataset = data_group.create_dataset(image_group_key, shape=(0,) + initial_img.shape, maxshape=max_image_shape, dtype='uint8')

        img_index = 0
        for p in prop:
            # action = servoing(rl_obs, p, last_reached)
            for i in range(2):
                action, done = servoing(rl_obs, p, last_reached, pid_params)
                rl_obs, reward, terminal, _, image_obs = env.step(action)
                image = image_obs['corner2'].cpu().numpy()
                actions.append(action)
                rewards.append(reward)
                new_prop.append(rl_obs["prop"].cpu().numpy())
                states.append(rl_obs["state"].cpu().numpy())

                img_dataset.resize((img_dataset.shape[0] + 1,) + image.shape)
                img_dataset[img_index] = image
                img_index += 1

                if done:
                    break

        print(f"Executed props: {action_key}")
        create_or_replace(data_group, action_key, actions)
        create_or_replace(data_group, reward_key, rewards)
        create_or_replace(data_group, prop_key, new_prop)
        create_or_replace(data_group, f'{prefix}/obs/state', states)    

        if img_dataset.shape[0] > len(actions):
            img_dataset.resize((len(actions),) + initial_img.shape)

        print("--------------------------------")


                    

def servoing(obs, waypoint, last_reached, pid_params):
    Kp, Ki, Kd = pid_params['Kp'], pid_params['Ki'], pid_params['Kd']
    error = torch.tensor(waypoint) - obs["prop"][:3].cpu()
    error_np = error.numpy()
    print("errp", error_np)
    
    done = False

    if 'integral' not in pid_params:
        pid_params['integral'] = np.zeros_like(error_np)
    if 'previous_error' not in pid_params:
        pid_params['previous_error'] = np.zeros_like(error_np)

    pid_params['integral'] += error_np
    derivative = error_np - pid_params['previous_error']
    pid_params['previous_error'] = error_np

    error_norm = np.linalg.norm(error_np)
    gripper_control = -1 if error_norm < 0.1 else 1  # Gripper control based on proximity

    if error_norm < 0.002:
        last_reached[0] = True  # Mark target as reached when close enough
        done = True
    # if last_reached[0]:
    control_action = Kp * error_np + Ki * pid_params['integral'] + Kd * derivative
    action = np.concatenate((control_action, [gripper_control]))
    # else:
        # action = np.array([0, 0, 0, gripper_control])  # No movement, only gripper control

    action = np.clip(action, -1, 1)  # Ensure action is within valid range

    return action, done

# Initialize PID parameters
pid_params = {
    'Kp': 15.0,
    'Ki': 0.1,
    'Kd': 0.01
}



    
def main(original_points, img_name, obj_pose):
    last_reached = [False]  # Initialize last_reached as a list to maintain state between calls
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
    save_hdf5(original_points, fitted_trajectory, noisy_trajectories, img_name, obj_pose, last_reached)


if __name__ == "__main__":

    import argparse
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process HDF5 dataset file.")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the HDF5 dataset file")
    parser.add_argument('--env_name', type=str, required=True, help="Save file")

    # Parse arguments
    args = parser.parse_args()

    dataset = args.dataset
    env_name = args.env_name
    save_file = f"{env_name}_teachermodel"

    print(f"\n ----> Running for dataset: {save_file} <---- \n")

    data = h5py.File(dataset, 'r')

    # group = data['data']
    group = data
    # demo_keys = ['demo_0', 'demo_1', 'demo_10'] 
    print(list(group.keys()))
    all_points = []
    for key in group.keys():
        demo = group[key]
        points = demo["obs/prop"][:, :3]

        state = demo["obs/state"]
        # Get last three columns
        if env_name == "DrawerOpen" or env_name == "ButtonPressWall" or env_name == "ButtonPressTopdownWall":
            obj_pose = state[0, 4:7]
        elif env_name == "CoffeeButton":
            obj_pose = state[0, 4:7] - np.array([.0, -.22, .3])
        else:
            obj_pose = state[0, -3:]  # Retrive obj pose from hdf5
        
        all_points.append(points)
        # main(points, image_name, key)
        main(points, key, obj_pose)
    # get_images_for_hdf5("trajectories.hdf5")  # Calling the function to capture and store images


data.close()  