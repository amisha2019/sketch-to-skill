import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_trajectory_points_heatmaps(data, point_type, task_type, output_dir):
    # data shape is (num_trajectories, num_points, 3)
    
    if point_type == 'starting':
        points = data[:, 0, :]  # Extract the first point of each trajectory
    elif point_type == 'ending':
        points = data[:, -1, :]  # Extract the last point of each trajectory
    elif point_type == 'trajectory':
        points = data.view(-1, 3)  # Flatten all points into a 2D array
    else:
        raise ValueError("point_type must be 'starting', 'ending', or 'trajectory'")
    
    # Separate x, y, and z coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    
    # Create subplots for three heatmaps
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # XY projection
    counts_xy, xedges, yedges, im1 = ax1.hist2d(x.numpy(), y.numpy(), bins=50, cmap='viridis')
    ax1.set_title(f'XY Projection ({point_type} points)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_aspect('equal', 'box')
    plt.colorbar(im1, ax=ax1, label='Density')
    
    # YZ projection
    counts_yz, yedges, zedges, im2 = ax2.hist2d(y.numpy(), z.numpy(), bins=50, cmap='viridis')
    ax2.set_title(f'YZ Projection ({point_type} points)')
    ax2.set_xlabel('Y')
    ax2.set_ylabel('Z')
    ax2.set_aspect('equal', 'box')
    plt.colorbar(im2, ax=ax2, label='Density')
    
    # ZX projection
    counts_zx, zedges, xedges, im3 = ax3.hist2d(z.numpy(), x.numpy(), bins=50, cmap='viridis')
    ax3.set_title(f'ZX Projection ({point_type} points)')
    ax3.set_xlabel('Z')
    ax3.set_ylabel('X')
    ax3.set_aspect('equal', 'box')
    plt.colorbar(im3, ax=ax3, label='Density')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f'{task_type}_trajectory_{point_type}_points_density.png')
    plt.savefig(filename)
    plt.close()

def visualize_traj_dist(data, task_type, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # plot heatmaps for starting points
    plot_trajectory_points_heatmaps(data, 'starting', task_type, output_dir)

    # Plot heatmaps for ending points
    plot_trajectory_points_heatmaps(data, 'ending', task_type, output_dir)

    # Plot heatmaps for all points
    plot_trajectory_points_heatmaps(data, 'trajectory', task_type, output_dir)

def get_points(data, point_type):
    x, y = [], []
    for img in data:
        if point_type == 'start':
            mask = (img[1, :, :] == 255.0) & (img[2, :, :] == 0.0) & (img[0, :, :] == 0.0)
        elif point_type == 'end':
            mask = (img[0, :, :] == 255.0) & (img[1, :, :] == 0.0) & (img[2, :, :] == 0.0)
        elif point_type == 'sketch':
            mask = (img[1, :, :] + img[2, :, :] + img[0, :, :] > 0.0)
        else:
            raise ValueError("Invalid point_type. Choose 'start', 'end', or 'sketch'.")
        
        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        y.append(y_coords)
        x.append(x_coords)
    return torch.cat(x), torch.cat(y)

def visualize_sketch_dist(data, task_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    point_types = [
        ('start', 'Starting Points', 'Greens'),
        ('end', 'Ending Points', 'Reds'),
        ('sketch', 'Sketch Points', 'Blues')
    ]

    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    fig.suptitle(f'Density Maps for {task_name}', fontsize=16)

    for idx, (point_type, title, cmap) in enumerate(point_types):
        x, y = get_points(data, point_type)
        counts, xedges, yedges, im = axs[idx].hist2d(x.numpy(), y.numpy(), bins=64, cmap=cmap)
        axs[idx].set_title(title)
        axs[idx].set_xlabel('X')
        axs[idx].set_ylabel('Y')
        axs[idx].invert_yaxis()  # Invert Y-axis to match image coordinates
        axs[idx].set_aspect('equal', 'box')
        plt.colorbar(im, ax=axs[idx], label='Density')

    plt.tight_layout()
    filename = os.path.join(output_dir, f'{task_name}_sketch_all_points_density.png')
    plt.savefig(filename)
    plt.close()

if __name__ == '__main__':
    img_size = 320
    # tasks = ["assembly", "boxclose", "ButtonPress", "coffeepush"]
    tasks = ['PegInsertSide',
            'PlateSlideBackSide',
            'Push',
            'PushBack',
            'PushWall',
            'PlateSlide',
            'PlateSlideSide',
            'ReachWall',
            'Soccer',
            'StickPush',
            'Sweep',
            'SweepInto',]
    
    tasks = ["ButtonPress"]
    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_real/'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    
    for task in tasks:
        print("Visualizing the task of ", task)

        sketches1_tensor = torch.load(root_path + f'{task}/{task}_sketches1_{img_size}.pt')
        sketches2_tensor = torch.load(root_path + f'{task}/{task}_sketches2_{img_size}.pt')
        exec(f"{task}_sketches_tensor = torch.cat([sketches1_tensor, sketches2_tensor], dim=0)")  
        visualize_sketch_dist(eval(f"{task}_sketches_tensor"), task, root_path+'/viz_sketch_density/')

        # exec(f"{task}_fitted_trajectories_tensor = torch.load(root_path + f'{task}_fitted_trajectories_50.pt')")
        # visualize_traj_dist(eval(f"{task}_fitted_trajectories_tensor"), task, root_path+'/viz_trajectory_density/')

    # print("Aggregating all tasks and visualizing...")
    # allTasks_sketches_tensor = torch.cat([assembly_sketches_tensor, boxclose_sketches_tensor, ButtonPress_sketches_tensor, coffeepush_sketches_tensor], dim=0)
    # visualize_sketch_dist(allTasks_sketches_tensor, 'allTasks')
    # allTasks_fitted_trajectories_tensor = torch.cat([assembly_fitted_trajectories_tensor, boxclose_fitted_trajectories_tensor, ButtonPress_fitted_trajectories_tensor, coffeepush_fitted_trajectories_tensor], dim=0)
    # visualize_traj_dist(allTasks_fitted_trajectories_tensor, 'allTasks')
