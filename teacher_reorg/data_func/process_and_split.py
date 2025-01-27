import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from tqdm import tqdm
from traj_bspline import get_trajectory_params_bspline
import time


def plot_props(props):
    print(props[:, 3])
    plt.figure()
    ax = plt.axes(projection='3d')
    idx = props[:, 3] > 0.5
    ax.scatter(props[idx, 0], props[idx, 1], props[idx, 2], c=props[idx, 3], cmap='viridis')
    ax.scatter(props[~idx, 0], props[~idx, 1], props[~idx, 2], c=props[~idx, 3], cmap='Reds')
    plt.savefig(f'test_figs/props.png')


def check_props(props):
    # check if the props which are larger than 0.5 are connected
    idx = props[:, 3] > 0.5
    first_false = np.where(~idx)[0][0]
    assert first_false == sum(idx), "The props which are larger than 0.5 are not connected"
    print(f"Checked prop, connectting points: {props[first_false, :3]}")
    # all the connecting points are the same
    # Checked prop, connectting points: [0.12059106 0.60245084 0.06801877]
    # obj1 = states[i][0, 4:7]
    # all the obj1s are: [0.13000136 0.59999996 0.01827762]


def save_image(image, path):
    image = Image.fromarray(image)
    image.save(path)


def project_trajectory(trajectory, extrinsic, K):
    trajectory = trajectory.T
    traj_2d = K @ extrinsic[:3, :] @ np.vstack((trajectory, np.ones(trajectory.shape[1])))
    traj_2d = traj_2d[:2, :] / traj_2d[2, :]
    return traj_2d.T

def get_sketch(traj_2d):
    color = (0, 255, 255)
    start_color = (0, 0, 255)
    end_color = (0, 255, 0)
    img = np.zeros((224, 224, 3), np.uint8)
    for i in range(len(traj_2d) - 1):
        p1 = (int(traj_2d[i, 0]), int(traj_2d[i, 1]))
        p2 = (int(traj_2d[i+1, 0]), int(traj_2d[i+1, 1]))
        cv2.line(img, p1, p2, color, 2)  # Yellow line
    
    cv2.circle(img, tuple(traj_2d[0].astype(int)), 2, start_color, -1)  # Red circle for start
    cv2.circle(img, tuple(traj_2d[-1].astype(int)), 2, end_color, -1) # Green circle for end
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

    
def generate_sketch(trajectory):
    # camera 1
    cam1_extrinsic = np.array([[-0.70710678,  0.70710678,  0.        , -0.49497475],
                               [-0.19245009, -0.19245009, -0.96225045,  0.28867513],
                               [-0.68041382, -0.68041382,  0.27216553, -1.18392004],
                               [ 0.        ,  0.        ,  0.        ,  1.        ]])
    cam1_K = np.array([[-270.39191899,    0.        ,   112.        ],
                       [   0.        , -270.39191899,   112.        ],
                       [   0.        ,    0.        ,     1.        ]])

    # camera 2
    cam2_extrinsic = np.array([[-0.54990133, -0.83318276,  0.05843818,  0.48395318],
                               [ 0.37620774, -0.30954914, -0.87329666,  0.40964644],
                               [ 0.74570521, -0.4582421,   0.48367129, -1.59310361],
                               [ 0.        ,  0.        ,  0.        ,  1.        ]])
    cam2_K = np.array([[-193.98969045,    0.        ,  112.        ],
                       [   0.        , -193.98969045,  112.        ],
                       [   0.        ,     0.       ,    1.        ]])

    trajectory_2d_cam1 = project_trajectory(trajectory, cam1_extrinsic, cam1_K)
    trajectory_2d_cam2 = project_trajectory(trajectory, cam2_extrinsic, cam2_K)
    return get_sketch(trajectory_2d_cam1), get_sketch(trajectory_2d_cam2)

def test_code(data_path):
    rgbs1 = torch.load(f'{data_path}/Assembly_rgbs1_224.pt')
    rgbs2 = torch.load(f'{data_path}/Assembly_rgbs2_224.pt')
    sketches1 = torch.load(f'{data_path}/Assembly_sketches1_224.pt')
    sketches2 = torch.load(f'{data_path}/Assembly_sketches2_224.pt')
    states = torch.load(f'{data_path}/Assembly_states_raw.pt')
    props = torch.load(f'{data_path}/Assembly_props_raw.pt')
    trajectories = torch.load(f'{data_path}/Assembly_trajectories_raw.pt')

    # for i in range(len(props)):
    #     idx = props[i][:, 3] > 0.5
    #     sketch1, sketch2 = generate_sketch(trajectories[i][idx])
    #     save_image(sketch1, f'test_figs/sketch1_{i}.png')
    #     save_image(sketch2, f'test_figs/sketch2_{i}.png')
    #     plot_props(props[i])
    #     check_props(props[i])

    obj_offset = []
    for i in range(len(props)):
        idx = props[i][:, 3] > 0.5
        cidx = sum(idx)
        print(props[i][cidx, :3], states[i][0, 4:7], props[i][cidx, :3] - states[i][0, 4:7])
        obj_offset.append(props[i][cidx, :3] - states[i][0, 4:7])
    obj_offset = np.array(obj_offset)
    print(f"obj_offset mean: {obj_offset.mean(axis=0)}, obj_offset std: {obj_offset.std(axis=0)}")
        
    # goal_offset = []
    # for i in range(len(props)):
    #     print(props[i][-1, :3], states[i][0, -3:], props[i][-1, :3] - states[i][0, -3:])
    #     goal_offset.append(props[i][-1, :3] - states[i][0, -3:])
    #     # breakpoint()
    # goal_offset = np.array(goal_offset)
    # print(f"goal_offset mean: {goal_offset.mean(axis=0)}, goal_offset std: {goal_offset.std(axis=0)}")
    
    # split is done
    # save the splited data, 
    # 1. save the rgbs, sketches, states, props, trajectories
    # 2. obtain fitted trajectories
    # 3. obtain fitted params

def load_raw_data(data_path):
    rgbs1_raw = torch.load(f'{data_path}/Assembly_rgbs1_224.pt')
    rgbs2_raw = torch.load(f'{data_path}/Assembly_rgbs2_224.pt')
    sketches1_raw = torch.load(f'{data_path}/Assembly_sketches1_224.pt')
    sketches2_raw = torch.load(f'{data_path}/Assembly_sketches2_224.pt')
    states_raw = torch.load(f'{data_path}/Assembly_states_raw.pt')
    props_raw = torch.load(f'{data_path}/Assembly_props_raw.pt')
    trajectories_raw = torch.load(f'{data_path}/Assembly_trajectories_raw.pt')

    sketches1 = []
    sketches2 = []
    rgbs1 = []
    rgbs2 = []
    states = []
    props = []
    trajectories = []
    for i in range(len(props_raw)):
        idx = props_raw[i][:, 3] > 0.5
        for cur_idx in [idx, ~idx]:
            sketch1, sketch2 = generate_sketch(trajectories_raw[i][cur_idx])
            sketches1.append(sketch1)
            sketches2.append(sketch2)
            rgbs1.append(rgbs1_raw[i].permute(1, 2, 0).numpy())
            rgbs2.append(rgbs2_raw[i].permute(1, 2, 0).numpy())
            states.append(states_raw[i][cur_idx])
            props.append(props_raw[i][cur_idx])
            trajectories.append(trajectories_raw[i][cur_idx])

    sketches1 = np.array(sketches1).astype(np.uint8)
    sketches2 = np.array(sketches2).astype(np.uint8)
    rgbs1 = np.array(rgbs1).astype(np.uint8)
    rgbs2 = np.array(rgbs2).astype(np.uint8)
    print(f"rgbs1 shape: {rgbs1.shape}, rgbs2 shape: {rgbs2.shape}")
    print(f"sketches1 shape: {sketches1.shape}, sketches2 shape: {sketches2.shape}")

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
    sketches1, sketches2, rgbs1, rgbs2, trajectories, props, states = load_raw_data(file_path)

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
    data_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new/Assembly_nonsplit"
    save_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new"
    # load_samples_and_save(data_path, save_path, "Assembly")
    
    test_code(data_path)