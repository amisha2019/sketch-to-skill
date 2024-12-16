import os
import time
import torch
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
from traj_bspline import get_trajectory_params_bspline
import cv2

def save_image(image, path):
    image = Image.fromarray(image)
    image.save(path)

def change_color(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    purple_lower = np.array([130, 70, 70])  # Tighter lower bound for purple
    purple_upper = np.array([150, 255, 255])  # Tighter upper bound for purple
    purple_mask = cv2.inRange(hsv_image, purple_lower, purple_upper)
    image[purple_mask > 0] = [0, 255, 0]  # Green
    return image

def save_images_for_test(f_data_obs, keys, save_path, idx):
    os.makedirs(save_path, exist_ok=True)
    for key in keys:
        image = f_data_obs[key][:].transpose(1,2,0)
        image = change_color(image)

        resized_image = cv2.resize(image, (64, 64))

        key = key.replace("_blank", "_image")
        save_image(image, f"{save_path}/demo_{idx}_{key}.png")
        save_image(resized_image, f"{save_path}/demo_{idx}_{key}_64.png")
    

def load_raw_data(file_path, num_samples=None):
    sketches1 = []
    sketches2 = []
    rgb_sketch1 = []
    rgb_sketch2 = []
    states = []
    props = []
    trajectories = []

    # Load the dataset
    with h5py.File(file_path, 'r') as f:
        print(f"============ Loading data from {file_path} =============")
        
        f_data = f['data']
        length = len(list(f_data.keys()))

        print(f"Length of the dataset: {length}")
        index = np.arange(length)
        index = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        
        if num_samples is not None:
            index = np.random.choice(length, num_samples//len(file_path), replace=False)
        for i in tqdm(index, desc="Loading data"):
            obs_group = f_data[f'demo_{i}']['obs']
            image_name_list = [
                'corner2_blank_sketch',  # (3, 320, 320)
                # 'corner2_image',  # (136, 3, 96, 96)
                # 'corner2_rgb_sketch',  # (3, 320, 320)
                'corner_blank_sketch',  # (3, 320, 320)
                # 'corner_image',  # (136, 3, 96, 96)
                # 'corner_rgb_sketch',  # (3, 320, 320)
                # 'eye_in_hand_image'  # (136, 3, 96, 96)
            ]
            ff_name = file_path.split('/')[-2]
            f_name = file_path.split('/')[-1].split('.')[0]
            savepath = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/hand_drawn_data/sketches_real/{ff_name}/{f_name}"
            save_images_for_test(obs_group, image_name_list, savepath, i)
            sketches1.append(change_color(obs_group['corner_blank_sketch'][:].transpose(1,2,0)))
            sketches2.append(change_color(obs_group['corner2_blank_sketch'][:].transpose(1,2,0)))
            prop = obs_group['prop']
            props.append(prop[:])
            trajectories.append(prop[:, :3])
            rgb_sketch1.append(change_color(obs_group['corner_rgb_sketch'][:].transpose(1,2,0)))
            rgb_sketch2.append(change_color(obs_group['corner2_rgb_sketch'][:].transpose(1,2,0)))
            states.append(f_data[f'demo_{i}']['states'][:])

    sketches1 = np.array(sketches1).astype(np.uint8)
    sketches2 = np.array(sketches2).astype(np.uint8)
    rgb_sketch1 = np.array(rgb_sketch1).astype(np.uint8)
    rgb_sketch2 = np.array(rgb_sketch2).astype(np.uint8)
    
    traj_len = np.array([len(cur_traj) for cur_traj in trajectories])
    print(f"Trajectory length max: {traj_len.max()}, min: {traj_len.min()}")
    
    # Filter out trajectories that are too short
    valid_idx = traj_len >= 30
    sketches1 = sketches1[valid_idx]
    sketches2 = sketches2[valid_idx]
    rgb_sketch1 = rgb_sketch1[valid_idx]
    rgb_sketch2 = rgb_sketch2[valid_idx]
    trajectories = [trajectories[i] for i in range(len(trajectories)) if valid_idx[i]]
    props = [props[i] for i in range(len(props)) if valid_idx[i]]
    states = [states[i] for i in range(len(states)) if valid_idx[i]]

    print(f"============== Loaded {len(sketches1)} images ==============")
    print(f"Sketch shape: {sketches1.shape}, {sketches2.shape}, {rgb_sketch1.shape}, {rgb_sketch2.shape}")
    print(f"Sketch min: {min(sketches1.min(), sketches2.min())}, Sketch max: {max(sketches1.max(), sketches2.max())}")
    print(f"RGB min: {min(rgb_sketch1.min(), rgb_sketch2.min())}, RGB max: {max(rgb_sketch1.max(), rgb_sketch2.max())}")
    
    return sketches1, sketches2, rgb_sketch1, rgb_sketch2, trajectories, props, states

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

def load_samples_and_save(file_path, save_path, raw_file_name, img_size=[64, 320], num_control_points=[10, 20, 50], num_samples=None):

    save_path = f"{save_path}/{raw_file_name}"
    os.makedirs(save_path, exist_ok=True)

    stime = time.time()
    # file_path = f"{file_path}/{file_name}/{file_name}.hdf5"
    file_names = [f.split('.')[0] for f in os.listdir(f"{file_path}/{raw_file_name}") if f.endswith('.hdf5')]
    for file_name in file_names:
        cur_file_path = f"{file_path}/{raw_file_name}/{file_name}.hdf5"
        print(f"Loading data from {cur_file_path}")
        sketches1, sketches2, rgb_sketch1, rgb_sketch2, trajectories, props, states = load_raw_data(cur_file_path, num_samples=num_samples) 
        continue

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

                    rgb_sketch1_tensor = resize_sketches_and_to_tensor(rgb_sketch1, cur_img_size)
                    rgb_sketch2_tensor = resize_sketches_and_to_tensor(rgb_sketch2, cur_img_size)
                    torch.save(rgb_sketch1_tensor, f'{save_path}/{file_name}_rgb_sketch1_{cur_img_size}.pt')
                    torch.save(rgb_sketch2_tensor, f'{save_path}/{file_name}_rgb_sketch2_{cur_img_size}.pt')
                    print(f"Saved {file_name}_rgb_sketch1_{cur_img_size}.pt and {file_name}_rgb_sketch2_{cur_img_size}.pt")
            else:
                raise ValueError("img_size must be a list or an integer")

        if num_control_points is not None:
            if isinstance(num_control_points, int):
                num_control_points = [num_control_points]
            if isinstance(num_control_points, list):
                for cur_num_control_points in num_control_points:
                    use_uniform_knots = False if cur_num_control_points == 50 else True
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
        torch.save(props, f'{save_path}/{file_name}_props_raw.pt')
        torch.save(states, f'{save_path}/{file_name}_states_raw.pt')
        
        print(f"Saved data to {save_path}")
        print(f"Time taken: {time.time() - stime:.2f} sec")


if __name__ == "__main__":
    file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_real"
    save_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_real"
    
    file_names = [
        # 'toast_press',
        'toast_pick_place'
    ]
    
    for f_name in file_names:
        load_samples_and_save(file_path, save_path, f_name)

