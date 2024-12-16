import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from traj_bspline import get_trajectory_params_bspline
import cv2
from tqdm import tqdm


def crop_and_resave(f_name, save_path):
    
    root_path = f'{save_path}/{f_name}'
    sketches1 = torch.load(f'{root_path}/{f_name}_sketches1_224.pt')
    sketches2 = torch.load(f'{root_path}/{f_name}_sketches2_224.pt')

    sketches1 = sketches1.permute(0, 2, 3, 1).numpy() / 255.0
    sketches2 = sketches2.permute(0, 2, 3, 1).numpy() / 255.0

    rgbs1 = torch.load(f'{root_path}/{f_name}_rgbs1_224.pt')
    rgbs2 = torch.load(f'{root_path}/{f_name}_rgbs2_224.pt')
    rgbs1 = rgbs1.permute(0, 2, 3, 1).numpy() / 255.0
    rgbs2 = rgbs2.permute(0, 2, 3, 1).numpy() / 255.0

    num_samples = len(sketches1)

    # apply center crop to sketches and rgbs
    # the size of sketches and rgbs are 224x224, we only keep the center 124x124
    # resize sketches and rgbs to 64x64
    cut_size = 45
    sketches1 = [cv2.resize(sketches1[i][cut_size:-cut_size, cut_size:-cut_size, :], (64, 64)) for i in tqdm(range(num_samples), desc="Processing sketches1")]
    sketches2 = [cv2.resize(sketches2[i][cut_size:-cut_size, cut_size:-cut_size, :], (64, 64)) for i in tqdm(range(num_samples), desc="Processing sketches2")]
    rgbs1 = [cv2.resize(rgbs1[i][cut_size:-cut_size, cut_size:-cut_size, :], (64, 64)) for i in tqdm(range(num_samples), desc="Processing rgbs1")]
    rgbs2 = [cv2.resize(rgbs2[i][cut_size:-cut_size, cut_size:-cut_size, :], (64, 64)) for i in tqdm(range(num_samples), desc="Processing rgbs2")]

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
    save_path = f'/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new/'
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
    
    save_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/demo_datasets"
    file_names = ['ButtonPress',
                'ButtonPressTopdownWall',
                'ButtonPressWall',
                'CoffeeButton',
                'DrawerOpen',
                'Reach',
                'ReachWall',]
    
    save_path = f'/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new/'
    file_names = ['Assembly_gradient',]
                    # 'BoxClose',
                    # 'ButtonPress',
                    # 'ButtonPressTopdownWall',
                    # # 'ButtonPressWall',
                    # # 'CoffeeButton',
                    # 'CoffeePush',
                    # 'DrawerOpen',
                    # 'Reach',
                    # 'ReachWall',]
    
    for f_name in file_names:
        crop_and_resave(f_name, save_path)
    