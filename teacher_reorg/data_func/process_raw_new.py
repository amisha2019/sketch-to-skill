import os
import time
import torch
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm
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


def load_raw_data(file_path, num_samples=None):
    sketches1 = []
    sketches2 = []
    rgbs1 = []
    rgbs2 = []
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
        
        if num_samples is not None:
            index = np.random.choice(length, num_samples//len(file_path), replace=False)
        for i in tqdm(index, desc="Loading data"):
            if True:
                image_name_list = [
                    "rgb_sketch",
                    "rgb_sketch2",
                    "corner",
                    "corner2"
                    ]
                save_images_for_test(f_data[f'demo_{i}/obs'], image_name_list, "assembly_gradient_test")
                breakpoint()
            # breakpoint()
            obs_group = f_data[f'demo_{i}']['obs']
            sketches1.append(obs_group['corner'][:])
            sketches2.append(obs_group['corner2'][:])
            prop = obs_group['prop']
            props.append(prop[:])
            trajectories.append(prop[:, :3])
            rgbs1.append(obs_group['rgb_sketch'][:])
            rgbs2.append(obs_group['rgb_sketch2'][:])
            states.append(obs_group['state'][:])

    sketches1 = np.array(sketches1).transpose(0, 2, 3, 1).astype(np.uint8)
    sketches2 = np.array(sketches2).transpose(0, 2, 3, 1).astype(np.uint8)
    rgbs1 = np.array(rgbs1).transpose(0, 2, 3, 1).astype(np.uint8)
    rgbs2 = np.array(rgbs2).transpose(0, 2, 3, 1).astype(np.uint8)
    
    traj_len = np.array([len(cur_traj) for cur_traj in trajectories])
    print(f"Trajectory length max: {traj_len.max()}, min: {traj_len.min()}")
    
    # Filter out trajectories that are too short
    valid_idx = traj_len >= 30
    sketches1 = sketches1[valid_idx]
    sketches2 = sketches2[valid_idx]
    rgbs1 = rgbs1[valid_idx]
    rgbs2 = rgbs2[valid_idx]
    trajectories = [trajectories[i] for i in range(len(trajectories)) if valid_idx[i]]
    props = [props[i] for i in range(len(props)) if valid_idx[i]]
    states = [states[i] for i in range(len(states)) if valid_idx[i]]

    print(f"============== Loaded {len(sketches1)} images ==============")
    print(f"Sketch shape: {sketches1.shape}, {sketches2.shape}, {rgbs1.shape}, {rgbs2.shape}")
    print(f"Sketch min: {min(sketches1.min(), sketches2.min())}, Sketch max: {max(sketches1.max(), sketches2.max())}")
    print(f"RGB min: {min(rgbs1.min(), rgbs2.min())}, RGB max: {max(rgbs1.max(), rgbs2.max())}")
    
    return sketches1, sketches2, rgbs1, rgbs2, trajectories, props, states

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

def load_samples_and_save(file_path, save_path, file_name, img_size=[64, 224], num_control_points=[20, 50], num_samples=None):

    save_path = f"{save_path}/{file_name}"
    os.makedirs(save_path, exist_ok=True)

    stime = time.time()
    # file_path = f"{file_path}/{file_name}_frame_stack_1_224x224_end_on_success/dataset.hdf5"
    file_path = f"{file_path}/{file_name}/dataset.hdf5"
    # /fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_224_new/Assembly_gradient/dataset.hdf5
    sketches1, sketches2, rgbs1, rgbs2, trajectories, props, states = load_raw_data(file_path, num_samples=num_samples) 

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
    torch.save(props, f'{save_path}/{file_name}_props_raw.pt')
    torch.save(states, f'{save_path}/{file_name}_states_raw.pt')
    
    print(f"Saved data to {save_path}")
    print(f"Time taken: {time.time() - stime:.2f} sec")


if __name__ == "__main__":
    # file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/anti_corruption_1000"
    file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/1000_our_tasks"
    # fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_224_new/Assembly_gradient
    file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_224_new/"
    save_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new"

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
                    'ReachWall',
                    'Soccer',
                    'StickPush',
                    'Sweep',
                    'SweepInto',]
    
    file_names = ['Assembly',
                    'BoxClose',
                    'ButtonPress',
                    'ButtonPressTopdownWall',
                    'ButtonPressWall',
                    'CoffeeButton',
                    'CoffeePush',
                    'DrawerOpen',
                    'Reach',
                    'ReachWall',]
    
    file_names = ['Assembly_gradient']
    
    for f_name in file_names:
        try:
            load_samples_and_save(file_path, save_path, f_name)
        except Exception as e:
            print(f"Error processing {f_name}: {e}")