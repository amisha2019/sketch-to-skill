import h5py
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize_samples(corner_images, resizes=None):
    if resizes is None:
        resizes = [112, 56, 28]
    num_samples = len(corner_images)
    fig, axes = plt.subplots(len(resizes)+1, num_samples, figsize=(3*num_samples, 6))

    for i in range(num_samples):
        axes[0, i].imshow(corner_images[i].squeeze())
        axes[0, i].set_title(f"Sample {i+1}: Corner Image")
        
        for j, resize in enumerate(resizes):
            resized_image = cv2.resize(corner_images[i].squeeze(), (resize, resize))
            axes[j+1, i].imshow(resized_image)
            axes[j+1, i].set_title(f"Sample {i+1}: Resized Corner Image {resize}")

    plt.tight_layout()
    plt.savefig('real_sketch_images.png')


def spectral_ordering(points, k=10):
    from scipy.spatial import distance_matrix
    
    dist_matrix = distance_matrix(points, points)
    k_nearest = np.argsort(dist_matrix, axis=1)[:, 1:k+1]
    
    n = len(points)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, k_nearest[i]] = 1
        A[k_nearest[i], i] = 1
    
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    fiedler_vector = eigenvectors[:, 1]
    return np.argsort(fiedler_vector)


def fit_trajectory(image):
    # find all unique colors from the image
    unique_colors = set()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            unique_colors.add(tuple(image[i, j]))
    breakpoint()
    # find all the pixels that are not black, image shape is (224, 224, 3)
    yellow_pixels = []
    green_pixels = []
    red_pixels = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == [255, 255, 0]:
                yellow_pixels.append([i, j])
            elif image[i, j] == [0, 255, 0]:
                green_pixels.append([i, j])
            elif image[i, j] == [255, 0, 0]:
                red_pixels.append([i, j])
    
    yellow_pixels = np.array(yellow_pixels)
    green_pixels = np.array(green_pixels)
    red_pixels = np.array(red_pixels)
    breakpoint()
    # Sort yellow pixels based on spectral_ordering method
    spectral_order_yellow = spectral_ordering(yellow_pixels)
    

if __name__ == '__main__':
    file_path = '/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_datasets/sketch_data_assembly.hdf5'
    num_samples = 6

    corner_images = []
    corner2_images = []
    all_obs = []

    # Load the dataset
    with h5py.File(file_path, 'r') as f:
        print(f"Loading data from {file_path}")
        length = len(list(f.keys()))
        index = random.sample(range(length), num_samples)

        for i in index:
            corner_images.append(f[f'demo_{i}/corner'][:])
            corner2_images.append(f[f'demo_{i}/corner2'][:])
            all_obs.append(f[f'demo_{i}/obs'][:])
        
    print(len(corner_images))
    print(len(corner2_images))
    print(len(all_obs))

    print("Shape:")
    print(corner_images[0].shape)
    print(corner2_images[0].shape)
    print(all_obs[0].shape)

    fit_trajectory(corner_images[0])

    visualize_samples(corner_images)