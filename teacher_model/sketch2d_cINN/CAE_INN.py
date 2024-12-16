import torch
import torch.nn as nn
import torch.optim as optim
import FrEIA.framework as Ff  # FrEIA for CINN framework
import FrEIA.modules as Fm  # FrEIA for GLOWCouplingBlock, PermuteRandom, etc.
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import zipfile
from io import BytesIO
import time
import numpy as np
from scipy import interpolate
from torchvision.utils import save_image
import os
from datetime import datetime
import random


parser = argparse.ArgumentParser(description="Run VAE-CINN training or evaluation")
parser.add_argument('--mode', type=str, required=True, choices=['train', 'eval'], help="Mode: train or eval")
args = parser.parse_args()

# Padding the MNIST images to 64x64
class PadTo64x64:
    def __call__(self, img):
        return transforms.functional.pad(img, (18, 18, 18, 18), fill=0)  # Padding 28x28 to 64x64

class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super(ConvEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Convolutional layers for encoding
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # Fully connected layers to generate mu and log_var
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)  # Adjust for 64x64 image
        self.fc_var = nn.Linear(256 * 8 * 8, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # Flatten before fully connected layers
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super(ConvDecoder, self).__init__()

        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)  # Adjust for 64x64 image

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (batch, 64, 16, 16)
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (batch, 32, 32, 32)
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # (batch, out_channels, 64, 64)
            nn.Sigmoid()
        )

    def forward(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 256, 8, 8)  # Reshape to start decoding (batch, 128, 8, 8)
        x = self.decoder(z)
        return x

class VAE_CINN(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, condition_dim: int,
                 num_control_points: int, degree: int, encoder, decoder):
        super(VAE_CINN, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_control_points = num_control_points
        self.degree = degree
        self.num_knots = num_control_points + degree + 1
        self.n_dim_total = num_control_points * 2
        print("!!!!! in model n_dim_total", self.n_dim_total)
        # Load encoder and decoder
        self.encoder = encoder
        self.decoder = decoder

        # Conditioning the CINN
        self.fc_condition = nn.Linear(latent_dim, condition_dim)

        # Build CINN
        self.cinn = self.build_inn(self.condition_dim, self.num_control_points, self.num_knots)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10, max=10)  # Clamp log_var for numerical stability
        std = torch.exp(0.5 * logvar) + 1e-6  # Add epsilon to avoid zero variance
        eps = torch.randn_like(std)
        return eps * std + mu


    def build_inn(self, condition_dim, num_control_points, num_knots):
        cond = Ff.ConditionNode(condition_dim)
        nodes = [Ff.InputNode(self.n_dim_total)]
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(16):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': self.subnet, 'clamp': 1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def subnet(self, ch_in, ch_out):
        return nn.Sequential(
            nn.Linear(ch_in, 512),
            nn.ReLU(),
            nn.Linear(512, ch_out)
        )

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decoder(z)
        return recon, mu, log_var


        # New method to generate trajectories
    def generate_trajectory(self, params, num_points=100):
        knots = torch.linspace(0, 1, self.num_knots).repeat(params.shape[0], 1).to(params.device)
        # Ensure boundary knots are repeated
        knots[:, :self.degree + 1] = 0
        knots[:, -self.degree - 1:] = 1

        control_points = params.reshape(-1, self.num_control_points, 2)

        trajectories = []
        for cp, k in zip(control_points, knots):
            tck = (k.cpu().numpy(), cp.t().cpu().numpy(), self.degree)
            t = np.linspace(0, 1, num_points)
            trajectory = interpolate.splev(t, tck)
            trajectories.append(np.array(trajectory).T)

        return np.array(trajectories)

# Custom Dataset class to load images from zip
class ZipImageDataset(Dataset):
    def __init__(self, zip_file_path, image_filenames, transform=None):
        self.zip_file_path = zip_file_path
        self.image_filenames = image_filenames
        self.transform = transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        with zipfile.ZipFile(self.zip_file_path, 'r') as archive:
            with archive.open(self.image_filenames[idx]) as file:
                image = Image.open(BytesIO(file.read())).convert('L')  # Convert to grayscale
                if self.transform:
                    image = self.transform(image)
        return image, self.image_filenames[idx]  # Return the original filename along with the image

# ------------------ Step 1: Pretraining on MNIST ------------------ #
visualization_path = '/fs/nexus-scratch/amishab/Teacher_student_RLsketch/sketch2d_cINN/mnist_resized_64/'
os.makedirs(visualization_path, exist_ok=True)

# Adjust the transform for MNIST to resize and pad
mnist_transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.Pad(18, fill=0),  # Padding with zeros to simulate background
    transforms.ToTensor(),        # Convert to Tensor
    # PadTo64x64()                  # Pad to 64x64
])

print("Loading MNIST dataset for pretraining...")
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
mnist_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)


for batch_idx, (data, _) in enumerate(mnist_loader):
    for i in range(data.size(0)):
        img_path = os.path.join(visualization_path, f"mnist_resized_{batch_idx}_{i}.png")
        save_image(data[i], img_path)
    if batch_idx == 3:  # Limit to visualize first few batches
        break
print(f"Resized MNIST images saved in {visualization_path}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
####################################test
# Pretraining loop on MNIST
# print("Starting pretraining on MNIST...")
# num_epochs_mnist = 5
# mnist_reconstructed_path = '/fs/nexus-scratch/amishab/Teacher_student_RLsketch/sketch2d_cINN/reconstructed_mnist/'
# os.makedirs(mnist_reconstructed_path, exist_ok=True)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# vae_cinn_mnist = VAE_CINN(
#     in_channels=1,  # For grayscale images like MNIST
#     latent_dim=128,  # Latent dimension size for VAE
#     condition_dim=10,  # Conditioning dimension
#     num_control_points=10,  # Number of control points for trajectory generation
#     degree=3,  # Degree for B-spline curves
#     encoder=ConvEncoder(in_channels=1, latent_dim=128),  # Encoder definition for MNIST
#     decoder=ConvDecoder(latent_dim=128, out_channels=1)  # Decoder definition for MNIST
# ).to(device)

# vae_cinn_mnist.apply(initialize_weights)


# optimizer_mnist = torch.optim.Adam(vae_cinn_mnist.parameters(), lr=1e-4)
# for epoch in range(num_epochs_mnist):
#     epoch_start_time = time.time()
#     for batch_idx, (data, _) in enumerate(mnist_loader):
#         data = data.to(device)

#         optimizer_mnist.zero_grad()
#         recon, mu, log_var = vae_cinn_mnist(data)
#         loss = nn.MSELoss()(recon, data)
#         if torch.isnan(loss):
#             print("NaN loss encountered, stopping...")
#             break
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(vae_cinn_mnist.parameters(), max_norm=1.0)  # Gradient clipping
#         optimizer_mnist.step()

#         # Save reconstructed images for each batch in the last epoch
#         if epoch == num_epochs_mnist - 1:
#             for i in range(recon.size(0)):
#                 recon_path = os.path.join(mnist_reconstructed_path, f"reconstructed_mnist_epoch_{epoch+1}_batch_{batch_idx+1}_img_{i+1}.png")
#                 save_image(recon[i], recon_path)

#     epoch_end_time = time.time()
#     print(f"MNIST Epoch {epoch + 1}/{num_epochs_mnist}, Loss: {loss.item()}, Time: {epoch_end_time - epoch_start_time:.2f}s")

# print(f"Reconstructed MNIST images saved in {mnist_reconstructed_path}")

############################################


# Define the model for MNIST pretraining
vae_cinn_mnist = VAE_CINN(
    in_channels=1,
    latent_dim=32,
    condition_dim=10,
    num_control_points=10,
    degree=3,
    encoder=ConvEncoder(in_channels=1, latent_dim=32),
    decoder=ConvDecoder(latent_dim=32, out_channels=1)
).to(device)

vae_cinn_mnist.apply(initialize_weights)
optimizer_mnist = torch.optim.Adam(vae_cinn_mnist.parameters(), lr=1e-4)

# Pretraining loop on MNIST
print("Starting pretraining on MNIST...")
num_epochs_mnist = 5
for epoch in range(num_epochs_mnist):
    epoch_start_time = time.time()
    for batch_idx, (data, _) in enumerate(mnist_loader):
        data = data.to(device)

        optimizer_mnist.zero_grad()
        recon, mu, log_var = vae_cinn_mnist(data)
        loss = nn.MSELoss()(recon, data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae_cinn_mnist.parameters(), max_norm=1.0)
        optimizer_mnist.step()

    epoch_end_time = time.time()
    print(f"MNIST Epoch {epoch + 1}/{num_epochs_mnist}, Loss: {loss.item()}, Time: {epoch_end_time - epoch_start_time:.2f}s")

torch.save(vae_cinn_mnist.state_dict(), 'vae_cinn_mnist_pretrained.pth')

# ------------------ Step 2: Fine-tuning on Custom Dataset ------------------ #

print("Loading custom dataset for fine-tuning...")

# Fine-tuning with 64x64 resized images
preprocess_64 = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# Load custom dataset for fine-tuning
zip_file_path = '/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_datasets/extracted_images.zip'
with zipfile.ZipFile(zip_file_path, 'r') as archive:
    image_filenames = [f for f in archive.namelist() if f.endswith(('.png', '.jpg', '.jpeg'))]

# Split dataset for training and evaluation
train_size = 1000
eval_size = 500
selected_image_filenames_train = image_filenames[:train_size]
selected_image_filenames_eval = image_filenames[train_size:train_size + eval_size]

zip_image_dataset_train = ZipImageDataset(zip_file_path, selected_image_filenames_train, transform=preprocess_64)
zip_image_dataloader_train = DataLoader(zip_image_dataset_train, batch_size=16, shuffle=True)
# vae_cinn_finetune.load_state_dict(torch.load('vae_cinn_mnist_pretrained.pth'))

vae_cinn_finetune = VAE_CINN(
    in_channels=1,
    latent_dim=32,
    condition_dim=10,
    num_control_points=10,
    degree=3,
    encoder=ConvEncoder(in_channels=1, latent_dim=32),
    decoder=ConvDecoder(latent_dim=32, out_channels=1)
).to(device)

vae_cinn_finetune.load_state_dict(torch.load('vae_cinn_mnist_pretrained.pth'))
save_dir = "./saved_models/vae_cinn_finetuning_eval/"
os.makedirs(save_dir, exist_ok=True)

# Define optimizer and fine-tuning loop
# vae_cinn_finetune.apply(initialize_weights)
optimizer_finetune = torch.optim.Adam(vae_cinn_finetune.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_finetune, mode='min', factor=0.1, patience=3, verbose=True)
num_epochs_finetune = 5
image_byte_list = []

start_time = time.time()

# Fine-tuning loop
for epoch in range(num_epochs_finetune):
    epoch_start_time = time.time()
    for batch_idx, (data, original_filenames) in enumerate(zip_image_dataloader_train):
        data = data.to(device)

        optimizer_finetune.zero_grad()
        recon, mu, log_var = vae_cinn_finetune(data)

        loss = nn.MSELoss()(recon, data)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae_cinn_mnist.parameters(), max_norm=1.0)
        optimizer_finetune.step()
        scheduler.step(loss)


        if batch_idx % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs_finetune}, Batch {batch_idx + 1}, Loss: {loss.item()}")

        # Save reconstructed images in the last epoch
        if epoch == num_epochs_finetune - 1:
            for i in range(recon.size(0)):
                img_buffer = BytesIO()
                save_image(recon[i], img_buffer, format='png')
                img_buffer.seek(0)
                original_image_name = os.path.basename(original_filenames[i])
                reconstructed_image_name = f"reconstructed_epoch_{epoch+1}_batch_{batch_idx+1}_img_{i+1}_{original_image_name}"
                image_byte_list.append((reconstructed_image_name, img_buffer.read()))

    epoch_end_time = time.time()
    print(f"Epoch {epoch + 1}/{num_epochs_finetune}, Loss: {loss.item()}, Time: {epoch_end_time - epoch_start_time:.2f}s")

    # Save the model after each epoch
    model_save_path_epoch = os.path.join(save_dir, f"vae_cinn_finetuned_epoch_{epoch + 1}.pth")
    torch.save(vae_cinn_finetune.state_dict(), model_save_path_epoch)
    print(f"Model saved after epoch {epoch + 1} to {model_save_path_epoch}")

# Save reconstructed images to a zip file after the last epoch
zip_save_path = '/fs/nexus-scratch/amishab/Teacher_student_RLsketch/sketch2d_cINN/reconstructed_images_finetuned_eval.zip'
with zipfile.ZipFile(zip_save_path, 'w') as zf:
    for image_name, image_bytes in image_byte_list:
        zf.writestr(image_name, image_bytes)
    print(f"Reconstructed images saved to {zip_save_path}")

print("Fine-tuning completed!")
