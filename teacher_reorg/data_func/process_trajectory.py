import time
import torch
from traj_bspline import get_trajectory_params_bspline


def fit_and_save_trajectories(root_path, file_name, num_control_points=20):
    stime = time.time()
    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/'
    trajectories = torch.load(root_path + f'{file_name}_trajectories.pt')

    params, fitted_trajectories = get_trajectory_params_bspline(trajectories, num_control_points=num_control_points)
    params_tensor = torch.from_numpy(params).float()
    fitted_trajectories_tensor = torch.from_numpy(fitted_trajectories).float()
    
    print(f"Trajectory shape: {trajectories.shape}")
    print(f"Params shape: {params_tensor.shape}")
    print(f"Fitted Trajectory shape: {fitted_trajectories.shape}")
    
    torch.save(params_tensor, f'{root_path}/{file_name}_params_{num_control_points}.pt')
    torch.save(fitted_trajectories_tensor, f'{root_path}/{file_name}_fitted_trajectories_{num_control_points}.pt')

    print(f"Saved data to {root_path}/{file_name}_params_{num_control_points}.pt")
    print(f"Time taken: {time.time() - stime:.2f} sec")


if __name__ == "__main__":
    # file_names = ["assembly", 
    #              "boxclose",
    #              "coffeepush",
    #              "ButtonPress"]
    
    file_names = ["assembly_new", 
                 "coffeepush_new"]
    
    root_path = '/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets/'
    for f_name in file_names:
        fit_and_save_trajectories(root_path, f_name, num_control_points=20)
