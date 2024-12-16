# import h5py
# import numpy as np
# # Open the HDF5 file
# file_path = '/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic/can/ph/96/processed_96_demo20_servoing.hdf5'
# # ["sideview", "frontview", "agentview", "robot0_eye_in_hand"]
# # -----------------------------------------------------------------------
# # Read hdf5
# # -----------------------------------------------------------------------
# with h5py.File(file_path, 'r') as f:
#     print(f"Number of demos: {len(f['data'])}")
#     print(f"obs keys: {f['data/demo_0/obs'].keys()}")
#     for demo in f['data'].keys():
#         # read the actions
#         actions = f[f'data/{demo}/actions'][:]
#         print(f"Actions shape: {actions.shape}")
#         # images = f[f'data/{demo}/obs/agentview_image'][:]
#         # print(f"Agent_view shape: {images.shape}")
#         images2 = f[f'data/{demo}/obs/robot0_eye_in_hand_image'][:]
#         print(f"robot0_eye_in_hand shape: {images2.shape}")
#         print(f"Demo length: {demo} : {len(actions)}")
#         print("env_args" in f["data"].attrs)
#         print("-------------------------------------------------")

import h5py
import numpy as np

# Open the HDF5 file
file_path = '/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic/can/ph/96s/processed_96_demo20_servoing.hdf5'
# -----------------------------------------------------------------------
# Read hdf5
# -----------------------------------------------------------------------
with h5py.File(file_path, 'r') as f:
    print(f"Number of demos: {len(f['data'])}")
    print(f"obs keys: {f['data/demo_0/obs'].keys()}")
    
    # Initialize lists to store max and min values for all actions
    max_actions = []
    min_actions = []

    for demo in f['data'].keys():
        # Read the actions
        actions = f[f'data/{demo}/actions'][:]
        print(f"Actions shape: {actions.shape}")
        
        # Collect max and min actions values
        max_actions.append(actions.max())
        min_actions.append(actions.min())

        # Optionally, print the shape of images if needed
        # images = f[f'data/{demo}/obs/agentview_image'][:]
        # print(f"Agent_view shape: {images.shape}")

        images2 = f[f'data/{demo}/obs/robot0_eye_in_hand_image'][:]
        print(f"robot0_eye_in_hand shape: {images2.shape}")
        print(f"Demo length: {demo} : {len(actions)}")
        print("env_args" in f["data"].attrs)
        print("-------------------------------------------------")
    
    # Print max and min of actions
    print(f"Max action in dataset: {max(max_actions)}")
    print(f"Min action in dataset: {min(min_actions)}")

