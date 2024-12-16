import torch
import numpy as np
from tqdm import tqdm
from distances import ChamferDistance, FrechetDistance
import matplotlib.pyplot as plt

def sanity_check_trajectory_fitting(root_path, f_name):
    threshold = 1e-4
    chamferDist = ChamferDistance()
    frechetDist = FrechetDistance()
    traj_gt = torch.load(f'{root_path}/{f_name}/{f_name}_trajectories_raw.pt')
    traj_dense = torch.load(f'{root_path}/{f_name}/{f_name}_fitted_trajectories_50.pt')
    traj_sparse = torch.load(f'{root_path}/{f_name}/{f_name}_fitted_trajectories_20.pt')
    print(f"traj_dense shape: {traj_dense.shape}")
    print(f"X range: {traj_dense[:,:,0].min():.4f} to {traj_dense[:,:,0].max():.4f}")
    print(f"Y range: {traj_dense[:,:,1].min():.4f} to {traj_dense[:,:,1].max():.4f}")
    print(f"Z range: {traj_dense[:,:,2].min():.4f} to {traj_dense[:,:,2].max():.4f}")
    
    for i in tqdm(range(len(traj_gt)), desc=f"Checking {f_name}", leave=False):
        dist_dense_chamfer = chamferDist(torch.tensor(traj_gt[i]).unsqueeze(0), traj_dense[i].unsqueeze(0))
        dist_sparse_chamfer = chamferDist(torch.tensor(traj_gt[i]).unsqueeze(0), traj_sparse[i].unsqueeze(0))
        if dist_dense_chamfer.item() > threshold:
            print(f"============== {f_name} {i} dist_dense_chamfer: {dist_dense_chamfer.item()}")
        if dist_sparse_chamfer.item() > threshold:
            print(f"************************** {f_name} {i} dist_sparse_chamfer: {dist_sparse_chamfer.item()}")
        # else:
        #     print(f"{f_name} {i} is good")


def sanity_check_goal_pose(root_path, f_name):
    traj_gt = torch.load(f'{root_path}/{f_name}/{f_name}_trajectories_raw.pt')
    state_gt = torch.load(f'{root_path}/{f_name}/{f_name}_states_raw.pt')
    for traj, state in zip(traj_gt, state_gt):
        # print(traj[-1], state[-1, -3:])
        print(f"{f_name} {np.linalg.norm(traj[-1] - state[-1, -3:]):.4f} {traj[-1] - state[-1, -3:]}")


def rescale_traj(traj, start, end):
    # traj: [bs, num_points, 3]
    # start: [bs, 3]
    # end: [bs, 3]
    # Calculate the current start and end points
    current_start = traj[:, 0, :]
    current_end = traj[:, -1, :]
    
    # Calculate the scaling factor
    current_range = current_end - current_start
    target_range = end - start
    scale = target_range / current_range
    
    # Apply scaling to all points except the first and last
    scaled_traj = traj.clone()
    scaled_traj = start.unsqueeze(1) + (traj - current_start.unsqueeze(1)) * scale.unsqueeze(1)

    return scaled_traj


def sanity_check_traj_rescaling(root_path, f_name):
    traj_gt = torch.load(f'{root_path}/{f_name}/{f_name}_fitted_trajectories_50.pt')
    len_traj = len(traj_gt)
    start = torch.zeros((len_traj, 3))
    end = torch.ones((len_traj, 3))
    rescaled_traj = rescale_traj(traj_gt, start, end)
    traj_gt = traj_gt.numpy()
    rescaled_traj = rescaled_traj.numpy()
    for i in range(len_traj):
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax.plot(traj_gt[i][:, 0], traj_gt[i][:, 1], traj_gt[i][:, 2], label='Original Trajectory')
        ax.plot(traj_gt[i][0, 0], traj_gt[i][0, 1], traj_gt[i][0, 2], 'ro', label='Start Point')
        ax.plot(traj_gt[i][-1, 0], traj_gt[i][-1, 1], traj_gt[i][-1, 2], 'go', label='End Point')
        ax.plot([0], [0], [0], 'bo', markersize=5, label='Origin (0,0,0)')
        ax.plot([1], [1], [1], 'mo', markersize=5, label='Point (1,1,1)')
        ax = fig.add_subplot(122, projection='3d')
        ax.plot(rescaled_traj[i][:, 0], rescaled_traj[i][:, 1], rescaled_traj[i][:, 2], label='Rescaled Trajectory')
        ax.plot(rescaled_traj[i][0, 0], rescaled_traj[i][0, 1], rescaled_traj[i][0, 2], 'ro', label='Start Point')
        ax.plot(rescaled_traj[i][-1, 0], rescaled_traj[i][-1, 1], rescaled_traj[i][-1, 2], 'go', label='End Point')
        ax.plot([0], [0], [0], 'bo', markersize=5, label='Origin (0,0,0)')
        ax.plot([1], [1], [1], 'mo', markersize=5, label='Point (1,1,1)')
        # ax.legend()
        plt.savefig(f'{f_name}_traj_rescaling_{i}.png')
        plt.close()


if __name__ == "__main__":
    data_path = "real"
    if data_path == "new":
        root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new/'
        file_names = ['Dissassemble',
                        'DoorOpen',
                        'DrawerClose',
                        'Hammer',
                        'PegInsertSide',
                        'PickPlace',
                        'PlateSlideBack',
                        'PlateSlideBackSide',
                        'PlateSlideSide',
                        'Push',
                        'PushBack',
                        'PushWall',
                        'Soccer',
                        'StickPush',
                        'Sweep',
                        'SweepInto',
                        'Assembly',
                        'BoxClose',
                        'ButtonPress',
                        'ButtonPressTopdownWall',
                        'CoffeePush',
                        'DrawerOpen',
                        'Reach',
                        'ReachWall',]
    elif data_path == "test":
        root_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/demo_datasets"
        file_names = ['ButtonPress',
                    'ButtonPressTopdownWall',
                    'ButtonPressWall',
                    'CoffeeButton',
                    'DrawerOpen',
                    'Reach',
                    'ReachWall',]
    elif data_path == "real":
        root_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_real"
        file_names = ["ButtonPress"]
        
    for file_name in file_names:
        print(f"Checking {file_name}")
        sanity_check_trajectory_fitting(root_path, file_name)
        # sanity_check_goal_pose(root_path, file_name)
        # sanity_check_traj_rescaling(root_path, file_name)
        breakpoint()