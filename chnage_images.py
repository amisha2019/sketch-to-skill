import h5py
import numpy as np

def process_observation(obs):
    """
    Process a single observation by permuting its dimensions.
    Converts images from [H, W, C] to [C, H, W].

    Args:
        obs (numpy array): Input observation array.

    Returns:
        numpy array: Permuted array.
    """
    return np.transpose(obs, (2, 0, 1))  # Permute dimensions

def permute_images_in_hdf5(file_path):
    """
    Permute all image data in the HDF5 file from [H, W, C] to [C, H, W].

    Args:
        file_path (str): Path to the HDF5 file to process.
    """
    with h5py.File(file_path, 'a') as f:
        for demo_key in f['data'].keys():
            print(f"Processing {demo_key}...")

            # Navigate to the 'obs' group for this demo
            obs_group_path = f"data/{demo_key}/obs"
            if obs_group_path not in f:
                print(f"Warning: No 'obs' group found for {demo_key}. Skipping.")
                continue

            obs_group = f[obs_group_path]

            for obs_key in obs_group.keys():
                print(f"Processing {demo_key}/{obs_key}...")
                data = obs_group[obs_key][:]
                
                # Ensure the data is 3D (e.g., [H, W, C])
                if len(data.shape) == 3 and data.shape[-1] in [3, 4]:  # Assume 3 or 4 channels (RGB or RGBA)
                    # Apply permutation
                    permuted_data = process_observation(data)
                    
                    # Replace the dataset with the permuted data
                    del obs_group[obs_key]  # Remove existing dataset
                    obs_group.create_dataset(obs_key, data=permuted_data)
                    print(f"Permuted and updated {demo_key}/{obs_key}.")
                else:
                    print(f"Skipping {demo_key}/{obs_key}, not a valid image dataset.")

    print(f"All images in {file_path} have been permuted.")

if __name__ == "__main__":
    # Path to the HDF5 file
    file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic/square/ph/processed_96_demo10.hdf5"
    permute_images_in_hdf5(file_path)
