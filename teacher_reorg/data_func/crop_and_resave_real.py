import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from traj_bspline import get_trajectory_params_bspline
import cv2
from tqdm import tqdm


def crop_and_resave(f_name, save_path):
    
    root_path = f'{save_path}/{f_name}'
    sketches1 = torch.load(f'{root_path}/{f_name}_sketches1_320.pt')
    sketches2 = torch.load(f'{root_path}/{f_name}_sketches2_320.pt')

    # check idx ranges that are non-zero
    non_zero_idx1 = (sketches1.sum(axis=(0,1)) != 0).nonzero()
    non_zero_idx2 = (sketches2.sum(axis=(0,1)) != 0).nonzero()
    print(f"sketches1 non-zero idx range: {non_zero_idx1.min(axis=0)[0]}, {non_zero_idx1.max(axis=0)[0]}")
    print(f"sketches2 non-zero idx range: {non_zero_idx2.min(axis=0)[0]}, {non_zero_idx2.max(axis=0)[0]}")
    breakpoint()

    sketches1 = sketches1.permute(0, 2, 3, 1).numpy() / 255.0
    sketches2 = sketches2.permute(0, 2, 3, 1).numpy() / 255.0
    print(f"sketches1 shape: {sketches1.shape}")
    print(f"sketches2 shape: {sketches2.shape}")

    rgbs1 = torch.load(f'{root_path}/{f_name}_rgbs1_320.pt')
    rgbs2 = torch.load(f'{root_path}/{f_name}_rgbs2_320.pt')
    rgbs1 = rgbs1.permute(0, 2, 3, 1).numpy() / 255.0
    rgbs2 = rgbs2.permute(0, 2, 3, 1).numpy() / 255.0

    num_samples = len(sketches1)

    # apply center crop to sketches and rgbs
    sketches1 = [cv2.resize(sketches1[i][120:280, 20:180, :], (64, 64)) for i in tqdm(range(num_samples), desc="Processing sketches1")]
    sketches2 = [cv2.resize(sketches2[i][120:280, 20:180, :], (64, 64)) for i in tqdm(range(num_samples), desc="Processing sketches2")]
    rgbs1 = [cv2.resize(rgbs1[i][120:280, 20:180, :], (64, 64)) for i in tqdm(range(num_samples), desc="Processing rgbs1")]
    rgbs2 = [cv2.resize(rgbs2[i][120:280, 20:180, :], (64, 64)) for i in tqdm(range(num_samples), desc="Processing rgbs2")]

    sketches1 = torch.from_numpy(np.array(sketches1)).permute(0, 3, 1, 2) * 255.0
    sketches2 = torch.from_numpy(np.array(sketches2)).permute(0, 3, 1, 2) * 255.0
    rgbs1 = torch.from_numpy(np.array(rgbs1)).permute(0, 3, 1, 2) * 255.0
    rgbs2 = torch.from_numpy(np.array(rgbs2)).permute(0, 3, 1, 2) * 255.0

    torch.save(sketches1, f'{root_path}/{f_name}_sketches1_64_cropped.pt')
    torch.save(sketches2, f'{root_path}/{f_name}_sketches2_64_cropped.pt')
    torch.save(rgbs1, f'{root_path}/{f_name}_rgbs1_64_cropped.pt')
    torch.save(rgbs2, f'{root_path}/{f_name}_rgbs2_64_cropped.pt')
    
    print(f"Processed for {f_name}")


if __name__ == "__main__":
    save_path = f'/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_real/'
    file_names = [
        'ButtonPress',
        # 'toast_press'  # no need to crop
    ]
    
    for f_name in file_names:
        crop_and_resave(f_name, save_path)
    