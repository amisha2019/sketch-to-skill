import numpy as np
import cv2
from PIL import Image
import torch
import os
import time
from traj_bspline import get_trajectory_params_bspline


def save_image(image, path, img_size=64):
    if img_size == 64:
        thickness = 2
        kernel = np.ones((thickness, thickness), np.uint8)
        image = cv2.dilate(image, kernel, iterations=1)
        # Resize image to 64x64
        image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
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


def generate_trajectory(traj_num, dist_type=0):
    # define two Gaussian distributions, one for the start point, one for the end point
    # sample the start and end points from the distributions
    # generate a trajectory between the start and end points
    # return the trajectory
    if dist_type == 0:
        start_mean = np.array([0.4, 1.0, 0.3])
        end_mean = np.array([-0.4, 0.3, 0.2])
    elif dist_type == 1:
        start_mean = np.array([0, 1.0, 0.3])
        end_mean = np.array([0, 0.3, 0.2])
    elif dist_type == 2:
        start_mean = np.array([0.2, 1.0, 0.3])
        end_mean = np.array([-0.2, 0.3, 0.2])
    else:
        raise ValueError("Invalid dist_type")
    starts = np.random.uniform(low=start_mean-0.05, high=start_mean+0.05, size=(traj_num, 3))
    ends = np.random.uniform(low=end_mean-0.05, high=end_mean+0.05, size=(traj_num, 3))
    trajectories = []
    for i in range(traj_num):
        # Generate control points between start and end
        num_control_points = 4
        control_points = np.zeros((num_control_points, 3))
        control_points[0] = starts[i]
        control_points[-1] = ends[i]
        
        # Add random offset to middle control points
        for j in range(1, num_control_points-1):
            t = j/(num_control_points-1)
            mid_point = (1-t)*starts[i] + t*ends[i]
            random_offset = np.random.normal(0, 0.1, 3) 
            control_points[j] = mid_point + random_offset
            
        # Generate trajectory using cubic spline interpolation
        t = np.linspace(0, 1, 50)
        trajectory = np.zeros((50, 3))
        for j in range(50):
            # De Casteljau's algorithm for cubic Bezier curve
            b = control_points.copy()
            for k in range(num_control_points-1):
                for l in range(num_control_points-1-k):
                    b[l] = (1-t[j])*b[l] + t[j]*b[l+1]
            trajectory[j] = b[0]
        trajectories.append(trajectory)
    return trajectories

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

def load_samples_and_save(save_path, file_name, img_size=[64, 224], num_control_points=[20, 50]):

    save_path = f"{save_path}/{file_name}"
    os.makedirs(save_path, exist_ok=True)

    trajectories = generate_trajectory(200, dist_type=0)
    trajectories += generate_trajectory(200, dist_type=1)
    trajectories = np.array(trajectories)
    sketches1 = []  
    sketches2 = []
    for i, traj in enumerate(trajectories):
        sketch1, sketch2 = generate_sketch(traj)
        sketches1.append(sketch1)
        sketches2.append(sketch2)
    sketches1 = np.array(sketches1)
    sketches2 = np.array(sketches2)
    breakpoint()

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


if __name__ == "__main__":
    # Trajectory x limits: -0.51, 0.51
    # Trajectory y limits: 0.37, 0.93
    # Trajectory z limits: 0.04, 0.43
    
    path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/hand_drawn_data/sketches_real/ablation"
    trajectories = generate_trajectory(10, dist_type=2)
    print(f"Generated {len(trajectories)} trajectories")
    print(trajectories[0].shape)
    print(np.array([traj[0] for traj in trajectories]))
    print(np.array([traj[-1] for traj in trajectories]))
    np.save(f"{path}/trajectories_raw.npy", trajectories)
    for i, traj in enumerate(trajectories):
        sketch1, sketch2 = generate_sketch(traj)
        save_image(sketch1, f"{path}/demo_{i}_corner_image_sketch_64.png")
        save_image(sketch2, f"{path}/demo_{i}_corner2_image_sketch_64.png")

    # save_path = "/fs/nexus-projects/Sketch_VLM_RL/teacher_model/peihong/sketch_datasets_new"
    # load_samples_and_save(save_path=save_path, file_name="ablation")
