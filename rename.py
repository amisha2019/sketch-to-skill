import h5py

def recursive_copy(old_group, new_group):
    """
    Recursively copy all datasets and groups from an old HDF5 group to a new group.

    Args:
        old_group (h5py.Group): Group to copy from.
        new_group (h5py.Group): Group to copy to.
    """
    for key, item in old_group.items():
        if isinstance(item, h5py.Group):
            # Create a new group in the destination and recurse
            new_subgroup = new_group.create_group(key)
            recursive_copy(item, new_subgroup)
        elif isinstance(item, h5py.Dataset):
            # Copy dataset
            new_group.create_dataset(key, data=item[:])
        else:
            print(f"Warning: Unknown item type for key {key}, skipping.")

def copy_and_rename_hdf5_keys(old_file_path, new_file_path):
    """
    Copies keys from an old HDF5 file to a new file under a 'data' group,
    renaming them sequentially.

    Args:
        old_file_path (str): Path to the old HDF5 file.
        new_file_path (str): Path to the new HDF5 file.
    """
    with h5py.File(old_file_path, 'r') as old_file:
        # Get all top-level keys
        old_keys = sorted(old_file.keys())
        print("Keys in the old file:")
        for key in old_keys:
            print(f"  {key}")

        with h5py.File(new_file_path, 'w') as new_file:
            # Create the `data` group in the new file
            data_group = new_file.create_group("data")

            print("\nCopying and renaming keys...")
            for i, old_key in enumerate(old_keys):
                new_key = f"demo_{i}"  # Sequential naming
                print(f"Processing {old_key} -> {new_key}...")
                new_group = data_group.create_group(new_key)
                
                # Recursively copy the contents
                recursive_copy(old_file[old_key], new_group)
                print(f"Copied {old_key} to {new_key}")

            print("\nKeys in the new file:")
            for key in data_group.keys():
                print(f"  {key}")









if __name__ == "__main__":
    # Paths to your old and new HDF5 files/fs/nexus-projects/Sketch_VLM_RL/teacher_model/asingh/open_loop+robosuite/can_open_loop_final.hdf5
    old_file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic/square/ph/test_open_loop_new.hdf5"
    new_file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic/square/ph/processed_224_demo20_servoing.hdf5"

    # Call the function to copy and rename keys
    copy_and_rename_hdf5_keys(old_file_path, new_file_path)






# import h5py

# def inspect_hdf5_structure(file_path):
#     """
#     Inspect the structure of an HDF5 file and print all groups and datasets.
#     """
#     def print_structure(name, obj):
#         print(name)

#     with h5py.File(file_path, 'r') as f:
#         print(f"Inspecting structure of {file_path}:")
#         f.visititems(print_structure)

# if __name__ == "__main__":
#     hdf5_file_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/datasets_robomimic/can/ph/test_open_loop_final.hdf5" # Replace with your HDF5 file path
#     inspect_hdf5_structure(hdf5_file_path)


