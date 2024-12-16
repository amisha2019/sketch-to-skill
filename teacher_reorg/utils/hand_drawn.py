import torch
import numpy as np
from PIL import Image
import h5py
import os

from data_func.dataloader_sketch import Thickening, normalize, standardize


def load_hand_draw_data(img_size):
    root_path = "/nfshomes/peihong/Documents/Teacher_student_RLsketch/teacher_model/sketch3d_cINN_vae/rgb_images"
    sketches1_names = [
        f"{root_path}/demo_0_corner_sketch.tiff", 
        f"{root_path}/demo_6_corner_sketch.tiff", 
    ]
    sketches2_names = [    
        f"{root_path}/demo_0_corner2_sketch.tiff",
        f"{root_path}/demo_6_corner2_sketch.tiff",
    ]
    sketches_names = sketches1_names + sketches2_names
    # load the sketches
    sketches = []
    thickening = Thickening(thickness=4)
    for sketch_path in sketches_names:
        # load the sketch
        sketch = normalize(np.array(Image.open(sketch_path).convert('RGB')))
        # Apply thickening to the sketch
        sketch = thickening(torch.from_numpy(sketch).permute(2, 0, 1)).permute(1, 2, 0).numpy()
        sketches.append(sketch)
    sketches = np.array(sketches) # (img_num, orig_img_size, orig_img_size, 3)

    # Resize the sketches to match the desired image size
    sketches = torch.nn.functional.interpolate(
        torch.from_numpy(sketches).permute(0, 3, 1, 2),
        size=(img_size, img_size),
        mode='bilinear',
        align_corners=False
    )   # (img_num, 3, img_size, img_size)

    # convert to tensor and normalize to [-0.5, 0.5]
    # sketches = torch.tensor(sketches).permute(0, 3, 1, 2).float() - 0.5
    
    # standardize
    data_mean = 0.0037
    data_std = 0.0472
    sketches = (sketches - data_mean) / data_std

    sketches1 = sketches[:len(sketches1_names)]
    sketches2 = sketches[len(sketches1_names):]

    return sketches1, sketches2


def load_hand_draw_data_new(f_name, use_traj_rescale=False, load_traj=False, num_stages=1):
    obj_offset = {
        "Assembly":  np.array([-0.00941031,  0.00245088,  0.04974115]), 
        "Assembly_gradient": np.array([-0.00941031,  0.00245088,  0.04974115]), 
        "Assembly_split": np.array([-0.00941031,  0.00245088,  0.04974115]), 
    }
    goal_offset = {
        "Assembly": np.array([ 0.11734733, -0.0013127,  0.06140099]),
        "Assembly_gradient": np.array([ 0.11734733, -0.0013127,  0.06140099]),
        "Assembly_split": np.array([ 0.11734733, -0.0013127,  0.06140099]),
    }
    root_path = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/hand_drawn_data/sketches/{f_name}"
    sketch1_names = ["demo_0_corner_image_sketch_64",
                     "demo_1_corner_image_sketch_64",
                     "demo_2_corner_image_sketch_64",
                     "demo_3_corner_image_sketch_64",
                     "demo_4_corner_image_sketch_64",
                    #  "demo_10_corner_image_sketch_64"
                     ]
    sketch2_names = ["demo_0_corner2_image_sketch_64",
                     "demo_1_corner2_image_sketch_64",
                     "demo_2_corner2_image_sketch_64",
                     "demo_3_corner2_image_sketch_64",
                     "demo_4_corner2_image_sketch_64",
                    #  "demo_10_corner2_image_sketch_64"
                     ]
    if num_stages == 2:
        sketch1_names += ["demo_0_corner_image_2_sketch_64",
                          "demo_1_corner_image_2_sketch_64",
                          "demo_2_corner_image_2_sketch_64",
                          "demo_3_corner_image_2_sketch_64",
                          "demo_4_corner_image_2_sketch_64",
                        #   "demo_10_corner_image_2_sketch_64"
                          ]
        sketch2_names += ["demo_0_corner2_image_2_sketch_64",
                          "demo_1_corner2_image_2_sketch_64",
                          "demo_2_corner2_image_2_sketch_64",
                          "demo_3_corner2_image_2_sketch_64",
                          "demo_4_corner2_image_2_sketch_64",
                        #   "demo_10_corner2_image_2_sketch_64"
                          ]
    sketches1 = []
    sketches2 = []
    for sketch1_name, sketch2_name in zip(sketch1_names, sketch2_names):
        sketch1 = Image.open(f"{root_path}/{sketch1_name}.png").convert('RGB')
        sketch1 = np.array(sketch1).transpose(2, 0, 1)
        sketches1.append(sketch1)
        sketch2 = Image.open(f"{root_path}/{sketch2_name}.png").convert('RGB')
        sketch2 = np.array(sketch2).transpose(2, 0, 1)
        sketches2.append(sketch2)
    sketches1 = torch.tensor(np.array(sketches1))  
    sketches2 = torch.tensor(np.array(sketches2))

    sketches1 = standardize(normalize(sketches1))
    sketches2 = standardize(normalize(sketches2))

    if load_traj or use_traj_rescale:
        starts = []
        ends = []
        trajs = []
        root_dir = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_96_new"
        f_name = "Assembly" if "Assembly" in f_name else f_name
        file_path = os.path.join(root_dir, f"{f_name}_frame_stack_1_96x96_end_on_success", "dataset.hdf5")
        with h5py.File(file_path, 'r') as f:
            # demo_group_key = ["demo_0", "demo_1", "demo_10"]
            # for i, demo_name in enumerate(demo_group_key):
            for i, sketch_name in enumerate(sketch1_names):
                demo_name = sketch_name.split("_corner")[0]
                demo_group = f['data'][demo_name]
                obs_group = demo_group['obs']
                state = obs_group['state']
                if num_stages == 1:
                    starts.append(state[0, :3])
                    ends.append(state[0, -3:] + goal_offset[f_name])
                elif num_stages == 2:
                    if "corner_image_sketch" in sketch_name:
                        starts.append(state[0, :3])
                        ends.append(state[0, 4:7] + obj_offset[f_name])
                    else:
                        starts.append(state[0, 4:7] + obj_offset[f_name])
                        ends.append(state[0, -3:] + goal_offset[f_name])
                    # print(f"gripper_init_pose: {states[0][:3]}, obj_pose_1: {states[0][4:7]}, obj_pose_2: {states[0][11:14]}, goal_pose: {states[0][-3:]} ******")
                if load_traj:
                    traj = obs_group['prop'][:, :3]
                    trajs.append(traj)
        starts = torch.tensor(np.array(starts))
        ends = torch.tensor(np.array(ends))

        if load_traj:
            return sketches1, sketches2, starts, ends, trajs
        else:
            return sketches1, sketches2, starts, ends
        
    return sketches1, sketches2, None, None, None


def load_hand_draw_data_robomimic(f_name, use_traj_rescale=False, load_traj=False, num_stages=1):
    root_path = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/hand_drawn_data/sketches/{f_name}"
    sketch1_names = ["demo_0_corner_image_sketch_64",
                     "demo_1_corner_image_sketch_64",
                     "demo_10_corner_image_sketch_64"]
    sketch2_names = ["demo_0_corner2_image_sketch_64",
                     "demo_1_corner2_image_sketch_64",
                     "demo_10_corner2_image_sketch_64"]
    if num_stages == 2:
        sketch1_names += ["demo_0_corner_image_2_sketch_64",
                          "demo_1_corner_image_2_sketch_64",
                          "demo_10_corner_image_2_sketch_64"]
        sketch2_names += ["demo_0_corner2_image_2_sketch_64",
                          "demo_1_corner2_image_2_sketch_64",
                          "demo_10_corner2_image_2_sketch_64"]
    sketches1 = []
    sketches2 = []
    for sketch1_name, sketch2_name in zip(sketch1_names, sketch2_names):
        sketch1 = Image.open(f"{root_path}/{sketch1_name}.png").convert('RGB')
        sketch1 = np.array(sketch1).transpose(2, 0, 1)
        sketches1.append(sketch1)
        sketch2 = Image.open(f"{root_path}/{sketch2_name}.png").convert('RGB')
        sketch2 = np.array(sketch2).transpose(2, 0, 1)
        sketches2.append(sketch2)
    sketches1 = torch.tensor(np.array(sketches1))  
    sketches2 = torch.tensor(np.array(sketches2))

    sketches1 = standardize(normalize(sketches1))
    sketches2 = standardize(normalize(sketches2))

    if load_traj or use_traj_rescale:
        starts = []
        ends = []
        trajs = []
        
        root_dir = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_bc_96_new"
        file_path = os.path.join(root_dir, f"{f_name}_frame_stack_1_96x96_end_on_success", "dataset.hdf5")
        with h5py.File(file_path, 'r') as f:
            # demo_group_key = ["demo_0", "demo_1", "demo_10"]
            # for i, demo_name in enumerate(demo_group_key):
            for i, sketch_name in enumerate(sketch1_names):
                demo_name = sketch_name.split("_corner")[0]
                demo_group = f['data'][demo_name]
                obs_group = demo_group['obs']
                state = obs_group['state']
                if num_stages == 1:
                    starts.append(state[0, :3])
                    ends.append(state[0, -3:])
                elif num_stages == 2:
                    if "corner_image_sketch" in sketch_name:
                        starts.append(state[0, :3])
                        ends.append(state[0, 4:7] + obj_offset[f_name])
                    else:
                        starts.append(state[0, 4:7] + obj_offset[f_name])
                        ends.append(state[0, -3:] + goal_offset[f_name])
                    # print(f"gripper_init_pose: {states[0][:3]}, obj_pose_1: {states[0][4:7]}, obj_pose_2: {states[0][11:14]}, goal_pose: {states[0][-3:]} ******")
                if load_traj:
                    traj = obs_group['prop'][:, :3]
                    trajs.append(traj)
        starts = torch.tensor(np.array(starts))
        ends = torch.tensor(np.array(ends))

        if load_traj:
            return sketches1, sketches2, starts, ends, trajs
        else:
            return sketches1, sketches2, starts, ends
        
    return sketches1, sketches2, None, None, None


def load_hand_draw_data_real(f_name, use_traj_rescale=False, load_traj=False):
    root_path = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/hand_drawn_data/sketches_real/{f_name}"
    if f_name == "ButtonPress":
        sample_num = 30
        sample_ids = np.arange(sample_num)
        substages = None
    elif f_name == "toast_press":
        sample_num = 26
        sample_ids = [0, 1, 2, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        substages = None
    elif f_name == "ablation":
        sample_num = 10
        sample_ids = np.arange(sample_num)
        substages = None

    sketches1 = []
    sketches2 = []
    for i in sample_ids:
        sketch1_name = f"demo_{i}_corner_image_sketch_64.png"
        sketch2_name = f"demo_{i}_corner2_image_sketch_64.png"
        sketch1 = Image.open(f"{root_path}/{sketch1_name}").convert('RGB')
        sketch1 = np.array(sketch1).transpose(2, 0, 1)
        sketches1.append(sketch1)
        sketch2 = Image.open(f"{root_path}/{sketch2_name}").convert('RGB')
        sketch2 = np.array(sketch2).transpose(2, 0, 1)
        sketches2.append(sketch2)
    sketches1 = torch.tensor(np.array(sketches1))  
    sketches2 = torch.tensor(np.array(sketches2))

    sketches1 = standardize(normalize(sketches1))
    sketches2 = standardize(normalize(sketches2))

    if use_traj_rescale:
        starts = []
        ends = []
        trajs = []
        if f_name == "ButtonPress":
            file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_real/ButtonPress/data_half.hdf5"
        elif f_name == "toast_press":
            file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_real/toast_press/toast_press.hdf5"
        
        if f_name == "ablation":
            trajs = np.load("/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/hand_drawn_data/sketches_real/ablation/trajectories_raw.npy")
            starts = [traj[0, :3] for traj in trajs]
            ends = [traj[-1, :3] for traj in trajs]
        else:
            with h5py.File(file_path, 'r') as f:
                for i in sample_ids:
                    demo_group = f['data'][f"demo_{i}"]
                    obs_group = demo_group['obs']
                    if f_name == "square":
                        state = demo_group['robot0_eef_pos']
                    else:
                        state = demo_group['state']
                    # breakpoint()
                    starts.append(state[0, :3])
                    ends.append(state[-1, :3])
                    if load_traj:
                        traj = obs_group['prop'][:, :3]
                        trajs.append(traj)
        starts = torch.tensor(np.array(starts))
        ends = torch.tensor(np.array(ends))

        if load_traj:
            return sketches1, sketches2, starts, ends, trajs
        else:
            return sketches1, sketches2, starts, ends
        
    return sketches1, sketches2


def load_hand_draw_data_real_pickplace(f_name, use_traj_rescale=False, load_traj=False):
    
    if f_name == "toast_pick_place":
        root_path = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/hand_drawn_data/sketches_real/{f_name}"
        sample_num = 30
        sample_ids = [0, 1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        substages = ["bread_pick", "bread_place"]
    elif f_name == "square":
        root_path = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/hand_drawn_data/sketches/{f_name}"
        sample_num = 50
        sample_ids = np.arange(sample_num)
        substages = ["part1", "part2"]
    elif f_name == "can":
        root_path = f"/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/hand_drawn_data/sketches/{f_name}"
        sample_num = 20
        sample_ids = np.arange(1, sample_num + 1)
        substages = ["part1", "part2"]
        
    sketches1 = []
    sketches2 = []
    for stage in substages:
        for i in sample_ids:
            sketch1_name = f"demo_{i}_corner_image_sketch_64.png"
            sketch2_name = f"demo_{i}_corner2_image_sketch_64.png"
            sketch1 = Image.open(f"{root_path}/{stage}/{sketch1_name}").convert('RGB')
            sketch1 = np.array(sketch1).transpose(2, 0, 1)
            sketches1.append(sketch1)
            sketch2 = Image.open(f"{root_path}/{stage}/{sketch2_name}").convert('RGB')
            sketch2 = np.array(sketch2).transpose(2, 0, 1)
            sketches2.append(sketch2)
    sketches1 = torch.tensor(np.array(sketches1))  
    sketches2 = torch.tensor(np.array(sketches2))

    sketches1 = standardize(normalize(sketches1))
    sketches2 = standardize(normalize(sketches2))

    if use_traj_rescale:
        starts = []
        ends = []
        trajs = []
        if f_name == "toast_pick_place":
            file_pathes = ["/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_real/toast_pick_place/bread_pick.hdf5",
                         "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_sketch_RL/demo_dataset_real/toast_pick_place/bread_place.hdf5"]
        elif f_name == "square":
            file_pathes = ["/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic/square/split/square_part1.hdf5",
                           "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic/square/split/square_part2.hdf5"]
        elif f_name == "can":
            file_pathes = ["/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic/can/split/can224_part1.hdf5",
                           "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic/can/split/can224_part2.hdf5"]
        for file_path in file_pathes:
            with h5py.File(file_path, 'r') as f:
                for i in sample_ids:
                    demo_group = f['data'][f"demo_{i}"]
                    obs_group = demo_group['obs']
                    if f_name == "square" or f_name == "can":
                        state = obs_group['robot0_eef_pos']
                    else:
                        state = demo_group['state']
                    # breakpoint()
                    starts.append(state[0, :3])
                    ends.append(state[-1, :3])
                    if load_traj:
                        if f_name == "square" or f_name == "can":
                            traj = obs_group['robot0_eef_pos'][:, :3]
                        else:
                            traj = obs_group['prop'][:, :3]
                        trajs.append(traj)
        starts = torch.tensor(np.array(starts))
        ends = torch.tensor(np.array(ends))

        if load_traj:
            return sketches1, sketches2, starts, ends, trajs
        else:
            return sketches1, sketches2, starts, ends
        
    return sketches1, sketches2


if __name__ == "__main__":
    sketches1, sketches2 = load_hand_draw_data(64)
    print(sketches1.shape)
    print(sketches2.shape)
    print(sketches1.dtype)
    print(sketches2.dtype)
