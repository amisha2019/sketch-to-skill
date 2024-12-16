from env.metaworld_wrapper import PixelMetaWorld
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import torch
from scipy.spatial.transform import Rotation as R
import cv2
# import h5py
import camera_param


def run_episode(images, all_obs, env, cfg):
    image_id = cfg.sketch_img_id   # 0: on the first frame, -1: on the last frame
    draw_interaction_markers = cfg.sketch_w_marker

    sensed_gripper_positions = []
    target_gripper_positions = []
    trajectory_2d = []

    id = 5
    fovy = env.model.cam_fovy[id]
    fov_rad = np.deg2rad(fovy)
    cam_ori = env.model.cam_mat0[id].reshape(3,3).T
    cam_pos = env.model.cam_pos[id]
    w = images[0].shape[0]
    h = images[0].shape[1]

    for i in range(len(all_obs)):
        # image, obs = zipped
        sensed_gripper_position = all_obs[i][3]
        try:
            target_gripper_position = all_obs[i+1][3]
        except IndexError:      # need to handle the last obs
            target_gripper_position = all_obs[i][3]
        sensed_gripper_positions.append(sensed_gripper_position)
        target_gripper_positions.append(target_gripper_position)
        
        # Project 3D to 2D
        obj_pos = all_obs[i][:3]
        d = cam_ori.dot(obj_pos - cam_pos)
        d = - d / d[2]
        f = h / 2 / np.tan(fov_rad / 2.0)
        bx = d[0] * f + w/2
        by = d[1] * f + h/2

        trajectory_2d.append([bx, by])

    trajectory_2d = np.array(trajectory_2d)
    output_image = generate_sketch(images, trajectory_2d, sensed_gripper_positions, target_gripper_positions, image_id=image_id, draw_interaction_markers=draw_interaction_markers)

    if cfg.sketch_save_gif:
        sketch_img_list = []
        for i in range(len(images)):
            sketch_img = generate_sketch(images[:i+1], trajectory_2d[:i+1], sensed_gripper_positions, target_gripper_positions, image_id=0, encode_temporal=False, draw_interaction_markers=draw_interaction_markers)
            sketch_img_list.append(sketch_img)
        return output_image, sketch_img_list

    return output_image


def plot_trajectory_on_image(img, trajectory_2d, encode_temporal=True, color=(0, 0, 255)):
    start_color = np.array([0, 255, 255])
    end_color = np.array([0, 124, 191])
    diff_color = end_color - start_color
    # color order in BGR, (0, 0, 255) for red
    for i in range(len(trajectory_2d) - 1):
        p1 = (int(trajectory_2d[i, 0]), int(trajectory_2d[i, 1]))
        p2 = (int(trajectory_2d[i+1, 0]), int(trajectory_2d[i+1, 1]))
        if encode_temporal:
            normalized_time_step = i / len(trajectory_2d) * 0.9 + 0.1
            color = np.array(start_color) + diff_color * normalized_time_step
            color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.line(img, p1, p2, color, 2)

    return img


def plot_interaction_markers_on_image(img, trajectory_2d, sensed_gripper_positions, target_gripper_positions):
    # Lists to store the key time steps for closing and opening the gripper
    closing_steps = []
    opening_steps = []
    gripper_opened = False
    epsilon = 0.5 

    # Interaction Markers - Iterate through the sensed and target gripper joint positions
    for i in range(len(sensed_gripper_positions)):
        sensed_position = sensed_gripper_positions[i]
        target_position = target_gripper_positions[i]
        delta = target_position - sensed_position

        if delta < 0 and target_position <= epsilon:        # Check for opening action
            if not gripper_opened:
                opening_steps.append(i)
                gripper_opened = True
        if delta > 0 and target_position > epsilon:          # Check for closing action
            if gripper_opened and (i == 0 or (i > 0 and (sensed_gripper_positions[i-1] - target_gripper_positions[i-1]) <= 0)):
                closing_steps.append(i)

    # Draw interaction markers on the first frame
    for step in opening_steps:
        p1 = (int(trajectory_2d[step, 0]), int(trajectory_2d[step, 1]))
        cv2.circle(img, p1, 5, (255, 0, 0), -1)  # Blue circles for opening action, color order in BGR

    for step in closing_steps:
        p1 = (int(trajectory_2d[step, 0]), int(trajectory_2d[step, 1]))
        cv2.circle(img, p1, 5, (0, 255, 0), -1)  # Green circles for closing action, color order in BGR

    return img


def generate_sketch(images, trajectory_2d, sensed_gripper_positions, target_gripper_positions, image_id=0, encode_temporal=True, draw_interaction_markers=True):
    # image_id = 0: generate sketch on the first frame
    # image_id = -1: generate sketch on the last frame
    
    img_orig = images[image_id]
    img = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
    blended_img = plot_trajectory_on_image(img, trajectory_2d, encode_temporal=encode_temporal, color=(0, 255, 255))  # Yellow color for trajectory, color order in BGR

    if draw_interaction_markers:
        blended_img = plot_interaction_markers_on_image(blended_img, trajectory_2d, sensed_gripper_positions, target_gripper_positions)

    blended_img = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
    return blended_img
