import h5py
import numpy as np
from teacher_reorg.data_func.traj_bspline import get_trajectory_params_bspline
from teacher_reorg.data_func.dataloader import AugTrajectory


def load_traj_from_raw_demo(file_name, num_traj_per_demo=10):
    root_dir = "/fs/nexus-projects/Sketch_VLM_RL/amishab/demo_dataset_bc_96_new/Reach_frame_stack_1_96x96_end_on_success"
    file_path = f"{root_dir}/dataset.hdf5"

    augment_trajectory = AugTrajectory(noise_scale=0.015, num_control_points=20, smoothness=0.05, p=1)

    all_trajs = []
    all_goal_pos = []

    # Load the dataset
    with h5py.File(file_path, 'r') as f:
        print(f"Loading data from {file_path}")

        data = f["data"]
        # num_demos = len(list(data.keys()))
        num_demos=3

        for i in range(num_demos):
            traj = data[f'demo_{i}/obs/prop'][:, :3]
            aug_trajs = []
            _, fitted_trajectories = get_trajectory_params_bspline([traj], num_control_points=20, use_uniform_knots=False)
            for j in range(num_traj_per_demo):
                # adding noise to the trajectory
                cur_aug_traj = augment_trajectory(fitted_trajectories[0]).numpy()
                aug_trajs.append(cur_aug_traj)  # shape: (num_traj_per_demo, num_points, 3)
            # 3 + 1 + 7 + 7 = 18
            # state = data[f'demo_{i}/obs/state'][0]
            # goal1 = data[f'demo_{i}/obs/state'][:, 4:7]  # shape: (3,)
            # goal2 = data[f'demo_{i}/obs/state'][0, 11:14]  # shape: (3,)
            goal_pos = data[f'demo_{i}/obs/state'][0, -3:]  # shape: (3,)

            all_trajs.append(np.array(aug_trajs))
            all_goal_pos.append(np.tile(goal_pos[np.newaxis, :], (num_traj_per_demo, 1)))
            print(all_trajs[-1].shape)
            print(all_goal_pos[-1].shape)

    all_trajs = np.concatenate(all_trajs, axis=0)   # shape: (num_traj_per_demo * num_demos, num_points, 3)
    all_goal_pos = np.concatenate(all_goal_pos, axis=0) # shape: (num_traj_per_demo * num_demos, 3)

    return all_trajs, all_goal_pos

if __name__ == "__main__":
    all_trajs, all_goal_pos = load_traj_from_raw_demo("assembly")
