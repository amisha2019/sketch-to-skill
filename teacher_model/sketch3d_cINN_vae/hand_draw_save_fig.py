import h5py
import numpy as np
from PIL import Image

data_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_data_assembly_rgb_without_sketch.hdf5"
# data_path = "/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_datasets/sketch_data_assembly_rgb.hdf5"


with h5py.File(data_path, 'r') as f:
    print(f"Loading data from {data_path}")
    length = len(list(f.keys()))
    for key in f.keys():
        print(key)
        data = f[key]
        # <KeysViewHDF5 ['corner', 'corner2', 'obs']>
        corner_data = np.array(data["corner"])
        image = Image.fromarray(corner_data)
        image = image.resize((300, 300))
        image.save(f"rgb_images/{key}_corner.png")

        corner2_data = np.array(data["corner2"])
        image = Image.fromarray(corner2_data)
        image = image.resize((300, 300))
        image.save(f"rgb_images/{key}_corner2.png")

    # for i in range(length):
    #     corner_images.append(f[f'demo_{i}/corner'][:])
    #     corner2_images.append(f[f'demo_{i}/corner2'][:])
    #     all_obs.append(f[f'demo_{i}/obs'][:])


    # Reading a TIFF image
    tiff_image_path = "path_to_your_tiff_image.tiff"
    tiff_image = Image.open(tiff_image_path)
    tiff_image = tiff_image.resize((300, 300))
    tiff_image.save("rgb_images/tiff_image_resized.png")