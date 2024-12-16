import h5py
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env.metaworld_wrapper import PixelMetaWorld
import time
import os

def visualize_results(original_points, model_points, servoring_points, gripper_init_pose, obj_pose, goal_pose, img_name, save_data_path):
    if servoring_points is None:
        rows, cols = 2, 2
    else:
        rows, cols = 2, 3
    fig = plt.figure(figsize=(8*cols, 8*rows))  # Adjust size to better fit more subplots

    # Create a colormap
    cmap = plt.get_cmap('viridis')
    
    # Original points
    ax11 = fig.add_subplot(rows, cols, 1, projection='3d')  # Row 1, Col 1
    ax11.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c=np.arange(len(original_points)), cmap=cmap, alpha=0.3, s=10)
    ax11.plot(original_points[:, 0], original_points[:, 1], original_points[:, 2], 'b-*', linewidth=2)
    ax11.plot(gripper_init_pose[0], gripper_init_pose[1], gripper_init_pose[2], 'go', markersize=20)
    # ax11.plot(obj_pose[0], obj_pose[1], obj_pose[2], 'ro', markersize=20)
    # apply offset to obj_pose
    # ax11.plot(goal_pose[0], goal_pose[1], goal_pose[2], 'r*', markersize=20)
    ax11.set_box_aspect((1, 1, 1))
    ax11.set_aspect('equal')
    ax11.set_title('Original Trajectory')
    ax11.set_xlabel('X')
    ax11.set_ylabel('Y')
    ax11.set_zlabel('Z')
    ax11.view_init(elev=30, azim=45)

    # Fitted trajectory
    ax12 = fig.add_subplot(rows, cols, 2, projection='3d')  # Row 1, Col 2
    ax12.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c=np.arange(len(original_points)), cmap=cmap, alpha=0.3, s=10)
    for point in model_points:
        ax12.plot(point[:, 0], point[:, 1], point[:, 2], '-*', linewidth=2)
    ax12.plot(gripper_init_pose[0], gripper_init_pose[1], gripper_init_pose[2], 'go', markersize=20)
    # ax12.plot(obj_pose[0], obj_pose[1], obj_pose[2], 'ro', markersize=20)
    # ax12.plot(goal_pose[0], goal_pose[1], goal_pose[2], 'r*', markersize=20)
    ax12.set_box_aspect((1, 1, 1))
    ax12.set_aspect('equal')
    ax12.set_title('Model Trajectory')
    ax12.set_xlabel('X')
    ax12.set_ylabel('Y')
    ax12.set_zlabel('Z')
    ax12.view_init(elev=30, azim=45)

    # Original points (without equal aspect)
    ax21 = fig.add_subplot(rows, cols, cols+1, projection='3d')  # Row 2, Col 1
    ax21.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c=np.arange(len(original_points)), cmap=cmap, alpha=0.3, s=10)
    ax21.plot(original_points[:, 0], original_points[:, 1], original_points[:, 2], 'b-*', linewidth=2)
    ax21.plot(gripper_init_pose[0], gripper_init_pose[1], gripper_init_pose[2], 'go', markersize=20)
    # ax21.plot(obj_pose[0], obj_pose[1], obj_pose[2], 'ro', markersize=20)
    # apply offset to obj_pose
    # ax21.plot(goal_pose[0], goal_pose[1], goal_pose[2], 'r*', markersize=20)
    ax21.set_title('Original Trajectory (Without Equal Aspect)')
    ax21.set_xlabel('X')
    ax21.set_ylabel('Y')
    ax21.set_zlabel('Z')
    ax21.view_init(elev=30, azim=45)

    # Fitted trajectory (without equal aspect)
    ax22 = fig.add_subplot(rows, cols, cols+2, projection='3d')  # Row 2, Col 2
    ax22.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c=np.arange(len(original_points)), cmap=cmap, alpha=0.3, s=10)
    for point in model_points:
        ax22.plot(point[:, 0], point[:, 1], point[:, 2], '-*', linewidth=2)
    ax22.plot(gripper_init_pose[0], gripper_init_pose[1], gripper_init_pose[2], 'go', markersize=20)
    # ax22.plot(obj_pose[0], obj_pose[1], obj_pose[2], 'ro', markersize=20)
    # ax22.plot(goal_pose[0], goal_pose[1], goal_pose[2], 'r*', markersize=20)
    ax22.set_title('Model Trajectory (Without Equal Aspect)')
    ax22.set_xlabel('X')
    ax22.set_ylabel('Y')
    ax22.set_zlabel('Z')
    ax22.view_init(elev=30, azim=45)

    # Servoing trajectory
    if servoring_points is not None:
        ax13 = fig.add_subplot(rows, cols, 3, projection='3d')  # Row 1, Col 3
        ax13.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c=np.arange(len(original_points)), cmap=cmap, alpha=0.3, s=10)
        for point in servoring_points:
            ax13.plot(point[:, 0], point[:, 1], point[:, 2], '-*', linewidth=2)
        ax13.plot(gripper_init_pose[0], gripper_init_pose[1], gripper_init_pose[2], 'go', markersize=20)
        # ax13.plot(obj_pose[0], obj_pose[1], obj_pose[2], 'ro', markersize=20)
        # ax13.plot(goal_pose[0], goal_pose[1], goal_pose[2], 'r*', markersize=20)
        ax13.set_box_aspect((1, 1, 1))
        ax13.set_aspect('equal')
        ax13.set_title('Servoing Trajectory')
        ax13.set_xlabel('X')
        ax13.set_ylabel('Y')
        ax13.set_zlabel('Z')
        ax13.view_init(elev=30, azim=45)

        ax23 = fig.add_subplot(rows, cols, cols+3, projection='3d')  # Row 2, Col 3
        ax23.scatter(original_points[:, 0], original_points[:, 1], original_points[:, 2], c=np.arange(len(original_points)), cmap=cmap, alpha=0.3, s=10)
        for point in servoring_points:
            ax23.plot(point[:, 0], point[:, 1], point[:, 2], '-*', linewidth=2)
        ax23.plot(gripper_init_pose[0], gripper_init_pose[1], gripper_init_pose[2], 'go', markersize=20)
        # ax23.plot(obj_pose[0], obj_pose[1], obj_pose[2], 'ro', markersize=20)
        # ax23.plot(goal_pose[0], goal_pose[1], goal_pose[2], 'r*', markersize=20)
        ax23.set_title('Servoing Trajectory (Without Equal Aspect)')
        ax23.set_xlabel('X')
        ax23.set_ylabel('Y')
        ax23.set_zlabel('Z')
        ax23.view_init(elev=30, azim=45)

    plt.tight_layout()
    os.makedirs(f"{save_data_path}", exist_ok=True)
    plt.savefig(f"{save_data_path}/{img_name}.png")
    plt.close()


def save_hdf5(env_name, model_points, demo_name, obj_pose, goal_pose, last_reached, save_file):
    # helper function to create or replace a dataset in the hdf5 file
    def create_or_replace(group, name, data):
        full_path = f'{group.name}/{name}'
        if full_path in f:
            del f[full_path]
        group.create_dataset(name, data=data)

    # create the environment for servoring
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
    # reset the environment
    rl_obs, image_obs = env.reset()
    
    # open the hdf5 file
    with h5py.File(f"{save_file}.hdf5", 'a') as f:
        prefix = f'{demo_name}'
        data_group = f.require_group('data')
        print(f"Saving {prefix} to HDF5 file")

        # set the object pose, TODO: set the gripper pose?
        # if env_name == "Reach":
        #     env.obj_rand_init(False)
        #     env.env.env.env.mw_set_goal_pose(obj_pose)
        #     env.env.env.env.mw_reset_model()
        # else:
        env.obj_rand_init(False)
        env.set_obj_pose_sktchRL(obj_pose, goal_pose)

        num_points = 20
        idx = np.linspace(3, len(model_points) - 10, num=num_points, dtype=int)
        if idx[-1] != len(model_points) - 1:
            idx = np.concatenate([idx, [len(model_points) - 1]])
        print(f"picked idx for model points: {idx}")
        prop = model_points[idx]
        # prop = model_points

        if len(np.where(prop[:, 3] == 0)[0]) > 0:
            first_zero = np.where(prop[:, 3] == 0)[0][0]
            prop[first_zero-5:first_zero, -1] = 0

        image_group_key = f'{prefix}/obs/corner2_image'
        action_key = f'{prefix}/actions'
        reward_key = f'{prefix}/rewards'
        prop_key = f'{prefix}/obs/prop'
        done_key = f'{prefix}/dones'

        actions = []
        rewards = []
        new_prop = []
        states = []
        dones = []

        # reset the environment
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
            for i in range(3):
                action, done = servoing(rl_obs, p, last_reached)
                rl_obs, reward, terminal, _, image_obs = env.step(action)
                image = image_obs['corner2'].cpu().numpy()
                actions.append(action)
                rewards.append(reward)
                new_prop.append(rl_obs["prop"].cpu().numpy())
                states.append(rl_obs["state"].cpu().numpy())
                dones.append(terminal)

                img_dataset.resize((img_dataset.shape[0] + 1,) + image.shape)
                img_dataset[img_index] = image
                img_index += 1

                if done or terminal:
                    break
                
            if terminal:
                break

        # check if dones are zeros except the last one
        if not np.all(np.array(dones[:-1]) == 0):
            print(f"dones are not all zeros except the last one")
            breakpoint()
        dones[-1] = True

        print(f"Executed props: {action_key}")
        create_or_replace(data_group, action_key, actions)
        create_or_replace(data_group, reward_key, rewards)
        create_or_replace(data_group, prop_key, new_prop)
        create_or_replace(data_group, f'{prefix}/obs/state', states)   
        create_or_replace(data_group, done_key, dones)

        if img_dataset.shape[0] > len(actions):
            img_dataset.resize((len(actions),) + initial_img.shape)

    print(f"Successfully Saved {demo_name} to {save_file}.hdf5")
    # print(f"env obj_init_pos: {env.env.env.obj_init_pos}")
    print(f"gripper_init_pose: {states[0][:3]}, obj_pose_1: {states[0][4:7]}, obj_pose_2: {states[0][11:14]}, goal_pose: {states[0][-3:]} ******")
    print(f"obj offset: {states[0][4:7] - obj_pose}")
    print(f"goal offset: {states[0][-3:] - goal_pose}")
    print("--------------------------------")
    
    servoring_points = np.array(new_prop)[:, :3]

    return servoring_points
                    

def servoing(obs, waypoint, last_reached):
    gripper = waypoint[3]
    waypoint = waypoint[:3]
    # Initialize PID parameters
    pid_params = {
        'Kp': 15.0,
        'Ki': 0.1,
        'Kd': 0.01
    }
    Kp, Ki, Kd = pid_params['Kp'], pid_params['Ki'], pid_params['Kd']
    error = torch.tensor(waypoint) - obs["prop"][:3].cpu()
    error_np = error.numpy()
    # print("errp", error_np)
    
    done = False

    if 'integral' not in pid_params:
        pid_params['integral'] = np.zeros_like(error_np)
    if 'previous_error' not in pid_params:
        pid_params['previous_error'] = np.zeros_like(error_np)

    pid_params['integral'] += error_np
    derivative = error_np - pid_params['previous_error']
    pid_params['previous_error'] = error_np

    error_norm = np.linalg.norm(error_np)
    # 1 for close, -1/0 for open
    gripper_control = -1 if gripper > 0.5 else 1
    gripper_control = 0 if error_norm > 0.1 else gripper_control
    # gripper_control = -1 if error_norm < 0.1 else 1  # Gripper control based on proximity

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

    
def main(env_name, demo_name, original_points, model_points, gripper_init_pose, obj_pose, goal_pose, goal_pose_offset, save_data_path, num_gen_per_demo, raw_demo_id):
    # model_points: 10x50x3, gripper_init_pose: 3
    model_points[:,:,:3] = model_points[:,:,:3] - model_points[:, 0:1, :3] + gripper_init_pose
    visualize_results(original_points, model_points, None, gripper_init_pose, obj_pose, goal_pose, f"{env_name}_{demo_name}", f"{save_data_path}_gen_{num_gen_per_demo}")
    # save_hdf5(original_points, fitted_trajectory, noisy_trajectories, img_name, obj_pose, last_reached)
    last_reached = [False]  # Initialize last_reached as a list to maintain state between calls
    servoring_points = []
    for i in range(num_gen_per_demo):
        name = f"demo_{raw_demo_id * num_gen_per_demo + i}"
        print(f"demo name {demo_name}, raw demo id: {raw_demo_id}, num gen per demo: {num_gen_per_demo}, name: {name}")
        cur_servoring_points = save_hdf5(env_name, model_points[i], name, obj_pose, goal_pose, last_reached, f"{save_data_path}_gen_{num_gen_per_demo}")
        servoring_points.append(cur_servoring_points)
    visualize_results(original_points, model_points[:num_gen_per_demo], servoring_points, gripper_init_pose, obj_pose, goal_pose, f"{env_name}_{demo_name}_servo", f"{save_data_path}_gen_{num_gen_per_demo}")
    

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process HDF5 dataset file.")
    # parser.add_argument('--env_name', type=str, required=True, help="Save file")
    parser.add_argument('--env_name', type=str, default="Assembly", help="Save file")
    parser.add_argument('--num_gen_per_demo', type=int, default=1, help="Number of generated trajectories per raw demo")
    args = parser.parse_args()

    raw_demo_data_paths = {
        "ButtonPress": "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_96_new/ButtonPress_frame_stack_1_96x96_end_on_success/dataset.hdf5",
        "ButtonPressTopdownWall": "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_96_new/ButtonPressTopdownWall_frame_stack_1_96x96_end_on_success/dataset.hdf5",
        "Reach": "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_96_new/Reach_frame_stack_1_96x96_end_on_success/dataset.hdf5",
        "ReachWall": "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_96_new/ReachWall_frame_stack_1_96x96_end_on_success/dataset.hdf5",
        "BoxClose": "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_96_new/BoxClose_frame_stack_1_96x96_end_on_success/dataset.hdf5",
        "Assembly": "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_96_new/Assembly_frame_stack_1_96x96_end_on_success/dataset.hdf5",
        "CoffeePush": "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_96_new/CoffeePush_frame_stack_1_96x96_end_on_success/dataset.hdf5",
        "DrawerOpen": "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_96_new/DrawerOpen_frame_stack_1_96x96_end_on_success/dataset.hdf5",
    }

    generated_data_paths = {
        "ButtonPress": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_0927_rescale/vae_mlp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_2024-09-27_06-44-04/generated_trajectory_ButtonPress_inference_hand_drawn.hdf5",
        "ButtonPressTopdownWall": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_0927_rescale_latent16/vae_mlp_ep200_bs128_dim16_cosine_lr0.00010_kld0.00010_aug_2024-09-27_17-43-00/generated_trajectory_ButtonPressTopdownWall_inference_hand_drawn.hdf5",
        "Reach": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_0927_rescale_latent16/vae_mlp_ep200_bs128_dim16_cosine_lr0.00010_kld0.00010_aug_2024-09-27_17-43-00/generated_trajectory_Reach_inference_hand_drawn.hdf5",
        "ReachWall": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_0927_rescale/vae_mlp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_2024-09-27_06-44-04/generated_trajectory_ReachWall_inference_hand_drawn.hdf5",
        "BoxClose": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_0927_rescale/vae_mlp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_2024-09-27_06-44-04/generated_trajectory_BoxClose_inference_hand_drawn.hdf5",
        # "Assembly": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_0927_rescale_latent16/vae_mlp_ep200_bs128_dim16_cosine_lr0.00010_kld0.00010_aug_2024-09-27_17-43-00/generated_trajectory_Assembly_inference_hand_drawn.hdf5",
        "Assembly": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_2sketch_Ablation_on_Assembly/Assembly_split_vae_mlp_2sketches_20cp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_rescale_novae_2024-11-26_21-24-24/generated_trajectory_Assembly_split_inference_hand_drawn.hdf5",
        "CoffeePush": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2_0927_rescale/vae_mlp_ep200_bs128_dim32_cosine_lr0.00100_kld0.00010_aug_2024-09-27_06-44-04/generated_trajectory_CoffeePush_inference_hand_drawn.hdf5",
        "DrawerOpen": "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/VAE_MLP_Both_NewData2/vae_mlp_ep200_bs128_dim32_cosine_lr0.001_kld0.0005_aug_2024-09-26_13-58-22 [somewhat better]/generated_trajectory_DrawerOpen_inference_hand_drawn.hdf5",
    }

    goal_pose_offset = {
        "ButtonPress": np.array([0, 0.1, 0]),
        "ButtonPressTopdownWall": np.array([0,  7.96326711e-05, -9.99999683e-02]),
        "Reach": np.array([0, 0, 0]),
        "ReachWall": np.array([0, 0, 0]),
        "BoxClose": np.array([0, 0, 0]),
        "Assembly": np.array([0.02999999, -0.19999999, -0.08066762]),
        "CoffeePush": np.array([0, 0, 0]),
        "DrawerOpen": np.array([0, 0, 0]),
    }

    raw_demo_data_path = raw_demo_data_paths[args.env_name]
    generated_data_path = generated_data_paths[args.env_name]

    save_path = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/raw_demo_servoring_data_assembly/{args.env_name}_split"
    save_data_path = f"{save_path}/{args.env_name}_teachermodel"
    num_gen_per_demo = args.num_gen_per_demo

    offsets = np.array([
        [-0.00309017, -0.00012879,  0.0528518 ],
        [-0.00324434, -0.00174951,  0.05477247],
        [-0.00407189,  0.0108846,   0.01559021]
    ])

    with h5py.File(raw_demo_data_path, 'r') as f_raw, h5py.File(generated_data_path, 'r') as f_gen:
        # keys = ["demo_0", "demo_1", "demo_10"]
        keys = ["demo_0", "demo_1","demo_2", "demo_3", "demo_4"]
        for raw_demo_id, demo_name in enumerate(keys):
            demo_group_gen = f_gen[demo_name]
            demo_group_raw = f_raw['data'][demo_name]

            original_points = demo_group_raw['obs']['prop'][:, :3]  # original_points: (num_points_raw, 3)
            model_points = demo_group_gen[:]  # model_points: (num_gen, num_points_gen, 3)

            gripper_init_pose = demo_group_raw['obs']['state'][0, :3]  # gripper_init_pose: (3,)
            obj_pose = demo_group_raw['obs']['state'][0, 4:7]  # obj_pose: (3,)
            goal_pose = demo_group_raw['obs']['state'][0, -3:]  # obj_pose: (3,)

            print("============================================")
            print(f"original_points: {original_points.shape}, model_points: {model_points.shape}, gripper_init_pose: {gripper_init_pose.shape}, obj_pose: {obj_pose.shape}, goal_pose: {goal_pose.shape}")
            print(f"gripper_init_pose: {gripper_init_pose}, obj_pose: {obj_pose}, goal_pose: {goal_pose}")
            print("--------------------------------")

            main(args.env_name, demo_name, original_points, model_points, gripper_init_pose, obj_pose, goal_pose, goal_pose_offset[args.env_name], save_data_path, num_gen_per_demo, raw_demo_id)
        