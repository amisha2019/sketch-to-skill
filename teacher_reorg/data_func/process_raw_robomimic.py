import torch
import numpy as np
import h5py
from PIL import Image
import time
from tqdm import tqdm
import os
from traj_bspline import get_trajectory_params_bspline


def save_image(image, path):
    image = Image.fromarray(image)
    image.save(path)


def save_images_for_test(f_data_obs, keys, save_path):
    os.makedirs(save_path, exist_ok=True)
    for key in keys:
        image = f_data_obs[key][:].transpose(1,2,0)
        if "blank" in key: #BGR2RGB
            image = image[:, :, ::-1]
        save_image(image, f"{save_path}/{key}.png")
    

def load_raw_data(file_path, num_samples=None, test=False):
    sketches1 = []
    sketches2 = []
    rgbs1 = []
    rgbs2 = []
    trajectories = []

    # Load the dataset
    with h5py.File(file_path, 'r') as f:
        print(f"Loading data from {file_path}")
        f_data = f['data']
        length = len(list(f_data.keys()))
        print(f"Length of the dataset: {length}")
        if "can" in file_path:
            index = np.arange(1, length)
        else:
            index = np.arange(0, length)
        if num_samples is not None:
            index = np.random.choice(length, num_samples//len(file_path), replace=False)
        print(f_data["demo_0"].keys())
        print(f_data["demo_0"]["obs"].keys())
        print(f_data["demo_0"]["sketches"].keys())
        for i in tqdm(index, desc="Loading data"):
            if test:
                image_name_list = [
                    'agentview_blank_sketch', 
                    'agentview_rgb_sketch', 
                    # 'birdview_blank_sketch', 
                    # 'birdview_rgb_sketch', 
                    'frontview_blank_sketch', 
                    'frontview_rgb_sketch', 
                    # 'sideview_blank_sketch', 
                    # 'sideview_rgb_sketch'
                    ]
                save_images_for_test(f_data[f'demo_{i}/sketches'], image_name_list, "robomimic_test")
                breakpoint()
            sketches1.append(f_data[f'demo_{i}/sketches']['agentview_blank_sketch'][:].transpose(1,2,0)[:, :, ::-1])
            rgbs1.append(f_data[f'demo_{i}/sketches']['agentview_rgb_sketch'][:].transpose(1,2,0))
            if "can" in file_path:
                sketches2.append(f_data[f'demo_{i}/sketches']['frontview_blank_sketch'][:].transpose(1,2,0)[:, :, ::-1])
                rgbs2.append(f_data[f'demo_{i}/sketches']['frontview_rgb_sketch'][:].transpose(1,2,0))
            elif "square" in file_path:
                sketches2.append(f_data[f'demo_{i}/sketches']['sideview_blank_sketch'][:].transpose(1,2,0)[:, :, ::-1])
                rgbs2.append(f_data[f'demo_{i}/sketches']['sideview_rgb_sketch'][:].transpose(1,2,0))
            else:
                raise ValueError("Unknown task")
            trajectories.append(f_data[f'demo_{i}/obs']["robot0_eef_pos"][:])

    sketches1 = np.array(sketches1)
    sketches2 = np.array(sketches2)
    rgbs1 = np.array(rgbs1)
    rgbs2 = np.array(rgbs2)
    if sketches1.shape[1] == 3:
        sketches1 = sketches1.transpose(0, 2, 3, 1)
        sketches2 = sketches2.transpose(0, 2, 3, 1)
    if rgbs1.shape[1] == 3:
        rgbs1 = rgbs1.transpose(0, 2, 3, 1)
        rgbs2 = rgbs2.transpose(0, 2, 3, 1)

    print(f"Loaded {len(sketches1)} images")
    print(f"Sketch shape: {sketches1.shape}")
    print(f"Sketch min: {min(sketches1.min(), sketches2.min())}, Sketch max: {max(sketches1.max(), sketches2.max())}")
    
    return sketches1, sketches2, rgbs1, rgbs2, trajectories

def resize_sketches_and_to_tensor(sketches, img_size):
    # sketches: numpy array of shape (num_samples, height, width, 3)
    # img_size: int
    # return: torch tensor of shape (num_samples, 3, img_size, img_size)
    if img_size == sketches.shape[1]:
        sketches_tensor = torch.from_numpy(sketches).permute(0, 3, 1, 2).float()
    else:
        sketches_resized = np.array([np.array(Image.fromarray(sketch).resize((img_size, img_size))) for sketch in sketches])
        sketches_tensor = torch.from_numpy(sketches_resized).permute(0, 3, 1, 2).float()
    return sketches_tensor

def load_samples_and_save(file_path, save_path, raw_file_name, img_size=[64, 224], num_control_points=[20, 50], num_samples=None):

    save_path = f"{save_path}/{raw_file_name}"
    os.makedirs(save_path, exist_ok=True)

    stime = time.time()
    file_names = [f.split('.')[0] for f in os.listdir(f"{file_path}/{raw_file_name}/split/") if f.endswith('.hdf5')]
    for file_name in file_names:
        cur_file_path = f"{file_path}/{raw_file_name}/split/{file_name}.hdf5"
        print(f"Loading data from {cur_file_path}")
        sketches1, sketches2, rgbs1, rgbs2, trajectories = load_raw_data(cur_file_path, num_samples=num_samples) 

        file_name = file_name.replace("224", "")

        # Resize sketches to img_size
        if img_size is not None:
            if isinstance(img_size, int):
                image_size = [img_size]
            if isinstance(img_size, list):
                for cur_img_size in img_size:
                    sketches1_tensor = resize_sketches_and_to_tensor(sketches1, cur_img_size)
                    sketches2_tensor = resize_sketches_and_to_tensor(sketches2, cur_img_size)
                    torch.save(sketches1_tensor, f'{save_path}/{file_name}_sketches1_{cur_img_size}.pt')
                    torch.save(sketches2_tensor, f'{save_path}/{file_name}_sketches2_{cur_img_size}.pt')
                    print(f"Saved {file_name}_sketches1_{cur_img_size}.pt and {file_name}_sketches2_{cur_img_size}.pt")

                    rgbs1_tensor = resize_sketches_and_to_tensor(rgbs1, cur_img_size)
                    rgbs2_tensor = resize_sketches_and_to_tensor(rgbs2, cur_img_size)
                    torch.save(rgbs1_tensor, f'{save_path}/{file_name}_rgbs1_{cur_img_size}.pt')
                    torch.save(rgbs2_tensor, f'{save_path}/{file_name}_rgbs2_{cur_img_size}.pt')
                    print(f"Saved {file_name}_rgbs1_{cur_img_size}.pt and {file_name}_rgbs2_{cur_img_size}.pt")
            else:
                raise ValueError("img_size must be a list or an integer")

        if num_control_points is not None:
            if isinstance(num_control_points, int):
                num_control_points = [num_control_points]
            if isinstance(num_control_points, list):
                for cur_num_control_points in num_control_points:
                    use_uniform_knots = True if cur_num_control_points == 20 else False
                    params, fitted_trajectories = get_trajectory_params_bspline(trajectories, cur_num_control_points, use_uniform_knots=use_uniform_knots)
                    if params is not None:
                        params_tensor = torch.from_numpy(params).float()
                        torch.save(params_tensor, f'{save_path}/{file_name}_params_{cur_num_control_points}.pt')
                        print(f"Saved {file_name}_params_{cur_num_control_points}.pt")
                    fitted_trajectories_tensor = torch.from_numpy(fitted_trajectories).float()
                    torch.save(fitted_trajectories_tensor, f'{save_path}/{file_name}_fitted_trajectories_{cur_num_control_points}.pt')
                    print(f"Saved {file_name}_fitted_trajectories_{cur_num_control_points}.pt")
            else:
                raise ValueError("num_control_points must be a list or an integer")

        torch.save(trajectories, f'{save_path}/{file_name}_trajectories_raw.pt')
            
        print(f"Saved data to {save_path}")
        print(f"Time taken: {time.time() - stime:.2f} sec")
        # Time taken: 302.55 sec


if __name__ == "__main__":
    file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic"
    save_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_robomimic"
    # file_names = ["assembly", 
    #              "boxclose",
    #              "coffeepush"]

    # file_names = ["ButtonPress"]

    file_names = [
        "square", 
        # "can",
        ]
    # "boxclose_new",

    for f_name in file_names:
        load_samples_and_save(file_path, save_path, f_name)
