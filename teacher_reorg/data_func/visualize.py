import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from traj_bspline import get_trajectory_params_bspline


def visualize_samples(sketches1, sketches2, original_points, fitted_trajectory, img_name, dense_fitted_trajectory=None, rgbs1=None, rgbs2=None):
    num_samples = len(sketches1)
    subplot_num = 3 if dense_fitted_trajectory is None else 5
    if rgbs1 is not None:
        subplot_num += 2
    fig = plt.figure(figsize=(3*num_samples, 3*subplot_num))

    # Create a colormap
    cmap = plt.get_cmap('viridis')

    for i in range(num_samples):
        ax1 = fig.add_subplot(subplot_num, num_samples, i+1)
        ax1.imshow(sketches1[i].squeeze())
        ax1.set_title(f"Sample {i+1}: Sketch 1")
        ax1.set_aspect('equal', 'box')

        ax2 = fig.add_subplot(subplot_num, num_samples, num_samples+i+1)
        ax2.imshow(sketches2[i].squeeze())
        ax2.set_title(f"Sample {i+1}: Sketch 2")
        ax2.set_aspect('equal', 'box')
        
        ax3 = fig.add_subplot(subplot_num, num_samples, 2*num_samples+i+1, projection='3d')
        ax3.scatter(original_points[i][:, 0], original_points[i][:, 1], original_points[i][:, 2], c=np.arange(len(original_points[i])), cmap=cmap, alpha=0.3)
        ax3.plot(fitted_trajectory[i][:, 0], fitted_trajectory[i][:, 1], fitted_trajectory[i][:, 2], 'r-', linewidth=2)
        ax3.set_title(f"Sample {i+1}: Fitted Trajectory")

        if dense_fitted_trajectory is not None:
            ax4 = fig.add_subplot(subplot_num, num_samples, 3*num_samples+i+1, projection='3d')
            ax4.scatter(original_points[i][:, 0], original_points[i][:, 1], original_points[i][:, 2], c=np.arange(len(original_points[i])), cmap=cmap, alpha=0.3)
            ax4.plot(dense_fitted_trajectory[i][:, 0], dense_fitted_trajectory[i][:, 1], dense_fitted_trajectory[i][:, 2], 'r-', linewidth=2)
            ax4.set_title(f"Sample {i+1}: Dense Fitted Trajectory")

            ax5 = fig.add_subplot(subplot_num, num_samples, 4*num_samples+i+1, projection='3d')
            ax5.plot(fitted_trajectory[i][:, 0], fitted_trajectory[i][:, 1], fitted_trajectory[i][:, 2], 'r-', linewidth=2)
            ax5.plot(dense_fitted_trajectory[i][:, 0], dense_fitted_trajectory[i][:, 1], dense_fitted_trajectory[i][:, 2], 'g-', linewidth=2)
            ax5.set_title(f"Sample {i+1}: Comparison")

        if rgbs1 is not None:
            ax6 = fig.add_subplot(subplot_num, num_samples, 5*num_samples+i+1)
            ax6.imshow(rgbs1[i].squeeze())
            ax6.set_title(f"Sample {i+1}: RGB 1")
            ax6.set_aspect('equal', 'box')

            ax7 = fig.add_subplot(subplot_num, num_samples, 6*num_samples+i+1)
            ax7.imshow(rgbs2[i].squeeze())
            ax7.set_title(f"Sample {i+1}: RGB 2")
            ax7.set_aspect('equal', 'box')

    plt.tight_layout()
    plt.savefig(f"{img_name}.png")
    plt.close()

def test_traj_fitting(f_name, model_type='bspline', squential=True):
    num_samples = 6
    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_real/'
    sketches1 = torch.load(f'{root_path}/{f_name}_sketches1_64.pt')
    sketches2 = torch.load(f'{root_path}/{f_name}_sketches2_64.pt')
    trajectories = torch.load(f'{root_path}/{f_name}_trajectories_raw.pt')

    if squential:
        idx = np.arange(num_samples)
    else:
        idx = np.random.choice(len(sketches1), num_samples, replace=False)
    sketches1 = sketches1[idx]
    sketches2 = sketches2[idx]
    trajectories = trajectories[idx]
    
    if model_type == 'bspline':
        params, fitted_trajectories = get_trajectory_params_bspline(trajectories, num_control_points=20)
    else:
        pass
        # params, fitted_trajectories = get_trajectory_params_seb(trajectories, num_basis=20)
    visualize_samples(sketches1, sketches2, trajectories, fitted_trajectories, model_type)

def sample_and_visualize(f_name, squential=True):
    num_samples = 6
    root_path = f'/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_real/{f_name}'
    sketches1 = torch.load(f'{root_path}/{f_name}_sketches1_64.pt')
    sketches2 = torch.load(f'{root_path}/{f_name}_sketches2_64.pt')
    trajectories = torch.load(f'{root_path}/{f_name}_trajectories_raw.pt')
    fitted_trajectories = torch.load(f'{root_path}/{f_name}_fitted_trajectories_20.pt')
    dense_fitted_trajectory = torch.load(f'{root_path}/{f_name}_fitted_trajectories_50.pt')
    breakpoint()

    if squential:
        idx = np.arange(num_samples)
    else:
        idx = np.random.choice(len(sketches1), num_samples, replace=False)
    sketches1 = sketches1[idx].permute(0, 2, 3, 1).to(int).numpy()
    sketches2 = sketches2[idx].permute(0, 2, 3, 1).to(int).numpy()
    # trajectories = trajectories[idx].numpy()
    trajectories = [trajectories[i] for i in idx]
    fitted_trajectories = fitted_trajectories[idx].numpy()
    dense_fitted_trajectory = dense_fitted_trajectory[idx].numpy()

    if os.path.exists(f'{root_path}/{f_name}_rgbs1_64_cropped.pt'):
        rgbs1 = torch.load(f'{root_path}/{f_name}_rgbs1_64_cropped.pt')
        rgbs2 = torch.load(f'{root_path}/{f_name}_rgbs2_64_cropped.pt')
        rgbs1 = rgbs1[idx].permute(0, 2, 3, 1).to(int).numpy()
        rgbs2 = rgbs2[idx].permute(0, 2, 3, 1).to(int).numpy()
    else:
        rgbs1, rgbs2 = None, None

    img_name = f"{root_path}/{f_name}_sample_cropped"
    img_name += "_sequential" if squential else "_random"
    visualize_samples(sketches1, sketches2, trajectories, fitted_trajectories, img_name, dense_fitted_trajectory, rgbs1, rgbs2)


if __name__ == "__main__":
    # file_names = ["assembly", 
    #              "boxclose",
    #              "coffeepush",
    #              "ButtonPress"]
    
    file_names = ["assembly_new", 
                 "coffeepush_new"]
    
    file_names = ["toast_press"]
    
    for f_name in file_names:
        sample_and_visualize(f_name, squential=False)
    