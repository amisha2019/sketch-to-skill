import zipfile
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import os

# Paths to the zip files
reconstructed_zip_path = '/fs/nexus-scratch/amishab/Teacher_student_RLsketch/sketch2d_cINN/reconstructed_images_finetuned_new.zip'
original_zip_path = '/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_datasets/extracted_images.zip'

# Names of the images we want to compare
reconstructed_image_names = [
    'reconstructed_epoch_5_batch_2_img_2_corner_301.png',
    'reconstructed_epoch_5_batch_2_img_3_corner_274.png',
    'reconstructed_epoch_5_batch_2_img_4_corner2_195.png',
    'reconstructed_epoch_5_batch_24_img_15_corner_12.png'
]

# Extract corresponding original image names by trimming the 'reconstructed_' part and using only the 'corner_' part
original_image_names = [
    'corner_301.png',
    'corner_274.png',
    'corner2_195.png',
    'corner_12.png'
]

# Function to load image from a zip file
def load_image_from_zip(zip_path, image_name):
    with zipfile.ZipFile(zip_path, 'r') as zf:
        with zf.open(image_name) as img_file:
            return Image.open(BytesIO(img_file.read()))

# Load the reconstructed and original images
reconstructed_images = [load_image_from_zip(reconstructed_zip_path, img_name) for img_name in reconstructed_image_names]
original_images = [load_image_from_zip(original_zip_path, img_name) for img_name in original_image_names]

# Plot the images in a 2x4 grid: 2 rows (reconstructed, original) and 4 columns for each image pair
fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns

# Plot each reconstructed image and its corresponding original image
for i in range(4):
    # Plot reconstructed images in the first row
    axes[0, i].imshow(reconstructed_images[i], cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title(f"Reconstructed: {reconstructed_image_names[i]}", fontsize=8)

    # Plot original images in the second row
    axes[1, i].imshow(original_images[i], cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title(f"Original: {original_image_names[i]}", fontsize=8)

# Adjust the layout and save the plot
plt.tight_layout()
output_image_path = './reconstructed_vs_original_collage.png'
plt.savefig(output_image_path, bbox_inches='tight')
plt.close()

print(f"Collage of reconstructed and original images saved at: {output_image_path}")
