import torch
import numpy as np
import h5py
from PIL import Image
import time
from tqdm import tqdm


def load_raw_data(file_path, num_samples=None):
    sketches1 = []
    sketches2 = []
    props = []
    trajectories = []
    if "new" in file_path:
        rgbs1 = []
        rgbs2 = []
        states = []

    # Load the dataset
    with h5py.File(file_path, 'r') as f:
        print(f"Loading data from {file_path}")
        length = len(list(f.keys()))
        print(f"Length of the dataset: {length}")
        index = np.arange(length)
        if num_samples is not None:
            index = np.random.choice(length, num_samples//len(file_path), replace=False)
        for i in tqdm(index, desc="Loading data"):
            sketches1.append(f[f'demo_{i}/corner'][:])
            sketches2.append(f[f'demo_{i}/corner2'][:])
            prop = f.get(f'demo_{i}/prop') if f.get(f'demo_{i}/obs') is None else f.get(f'demo_{i}/obs')
            props.append(prop[:])
            trajectories.append(prop[:, :3])
            if "new" in file_path:
                rgbs1.append(f[f'demo_{i}/rgb'][:])
                rgbs2.append(f[f'demo_{i}/rgb2'][:])
                states.append(f[f'demo_{i}/state'][:])

    sketches1 = np.array(sketches1)
    sketches2 = np.array(sketches2)
    if sketches1.shape[1] == 3:
        sketches1 = sketches1.transpose(0, 2, 3, 1)
        sketches2 = sketches2.transpose(0, 2, 3, 1)
    for i, cur_prop in enumerate(props):
        if cur_prop.shape != props[0].shape:
            print(f"Warning: Inconsistent shape detected for demo_{i}. Expected {props[0].shape}, but got {cur_prop.shape}")
            breakpoint()
    props = np.array(props)
    trajectories = np.array(trajectories)

    if "new" in file_path:
        rgbs1 = np.array(rgbs1)
        rgbs2 = np.array(rgbs2)
        if rgbs1.shape[1] == 3:
            rgbs1 = rgbs1.transpose(0, 2, 3, 1)
            rgbs2 = rgbs2.transpose(0, 2, 3, 1)
        states = np.array(states)

    print(f"Loaded {len(sketches1)} images")
    print(f"Sketch shape: {sketches1.shape}")
    print(f"Sketch min: {min(sketches1.min(), sketches2.min())}, Sketch max: {max(sketches1.max(), sketches2.max())}")
    
    if "new" in file_path:
        return sketches1, sketches2, props, trajectories, rgbs1, rgbs2, states
    else:
        return sketches1, sketches2, props, trajectories, None, None, None

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

def load_samples_and_save(file_path, save_path, file_name, img_size=[64, 224], num_samples=None):

    stime = time.time()
    if "new" in file_name:
        file_path = f"{file_path}/sketch_{file_name}.hdf5"
    else:
        file_path = f"{file_path}/sketch_data_{file_name}.hdf5"
    sketches1, sketches2, props, trajectories, rgbs1, rgbs2, states = load_raw_data(file_path, num_samples=num_samples) 

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

                if "new" in file_name:
                    rgbs1_tensor = resize_sketches_and_to_tensor(rgbs1, cur_img_size)
                    rgbs2_tensor = resize_sketches_and_to_tensor(rgbs2, cur_img_size)
                    torch.save(rgbs1_tensor, f'{save_path}/{file_name}_rgbs1_{cur_img_size}.pt')
                    torch.save(rgbs2_tensor, f'{save_path}/{file_name}_rgbs2_{cur_img_size}.pt')
                    print(f"Saved {file_name}_rgbs1_{cur_img_size}.pt and {file_name}_rgbs2_{cur_img_size}.pt")
        else:
            raise ValueError("img_size must be a list or an integer")

    props_tensor = torch.from_numpy(props).float()
    trajectories_tensor = torch.from_numpy(trajectories).float()
    torch.save(props_tensor, f'{save_path}/{file_name}_props.pt')
    torch.save(trajectories_tensor, f'{save_path}/{file_name}_trajectories.pt')

    if "new" in file_name:
        states_tensor = torch.from_numpy(states).float()
        torch.save(states_tensor, f'{save_path}/{file_name}_states.pt')
    
    print(f"Saved data to {save_path}")
    print(f"Time taken: {time.time() - stime:.2f} sec")
    # Time taken: 302.55 sec


if __name__ == "__main__":
    file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_datasets"
    save_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets"
    # file_names = ["assembly", 
    #              "boxclose",
    #              "coffeepush"]

    # file_names = ["ButtonPress"]

    file_names = ["assembly_new", 
                 "coffeepush_new"]
    # "boxclose_new",

    for f_name in file_names:
        load_samples_and_save(file_path, save_path, f_name)
