import time
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
import os
import zipfile
from torchvision.utils import save_image
import io  # For in-memory byte I/O
import time



class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int):
        super(ConvEncoder, self).__init__()
        self.latent_dim = latent_dim

        # Convolutional layers for encoding
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU()
        )
        # Fully connected layers to generate mu and log_var
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_var = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)  # Flatten before fully connected layers
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int, out_channels: int):
        super(ConvDecoder, self).__init__()

        self.decoder_input = nn.Linear(latent_dim, 128 * 4 * 4)

        self.decoder = nn.Sequential(
            # First transposed conv to go from 4x4 to 7x7
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),  # (batch, 64, 7, 7)
            nn.ReLU(),
            # Second transposed conv to go from 7x7 to 14x14
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # (batch, 32, 14, 14)
            nn.ReLU(),
            # Third transposed conv to go from 14x14 to 28x28 (correct output size)
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),  # (batch, out_channels, 28, 28)
            nn.Sigmoid()  # Output between 0 and 1 for image reconstruction
        )

    def forward(self, z):
        z = self.decoder_input(z)
        z = z.view(-1, 128, 4, 4)  # Reshape to start decoding (batch, 128, 4, 4)
        x = self.decoder(z)
        return x


class VAE_CINN(nn.Module):
    def __init__(self, in_channels: int, latent_dim: int, condition_dim: int,
                 num_control_points: int, degree: int, pretrained_encoder_path=None, pretrained_decoder_path=None):
        super(VAE_CINN, self).__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_control_points = num_control_points
        self.degree = degree
        self.num_knots = num_control_points + degree + 1
        self.n_dim_total = num_control_points * 2

        # Load convolutional encoder and decoder
        self.encoder = ConvEncoder(in_channels, latent_dim)
        self.decoder = ConvDecoder(latent_dim, in_channels)  # in_channels -> out_channels in decoder

        # Load pre-trained weights if provided
        if pretrained_encoder_path:
            self.encoder.load_state_dict(torch.load(pretrained_encoder_path))
        if pretrained_decoder_path:
            self.decoder.load_state_dict(torch.load(pretrained_decoder_path))

        # Conditioning the CINN
        self.fc_condition = nn.Linear(latent_dim, condition_dim)

        # Build CINN
        self.cinn = self.build_inn(self.condition_dim, self.num_control_points, self.num_knots)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def build_inn(self, condition_dim, num_control_points, num_knots):
        cond = Ff.ConditionNode(condition_dim)
        nodes = [Ff.InputNode(num_control_points * 2 + num_knots)]
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


import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import FrEIA.framework as Ff  # FrEIA for CINN framework
import FrEIA.modules as Fm  # FrEIA for GLOWCouplingBlock, PermuteRandom, etc.
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import zipfile
from io import BytesIO
import os
import zipfile
from torchvision.utils import save_image
import io  # For in-memory byte I/O

# ConvEncoder and ConvDecoder classes remain unchanged

# VAE_CINN class remains unchanged

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
        return image, image


preprocess = transforms.Compose([
    transforms.Resize((28, 28)),   # Resize the images to 28x28
    # transforms.RandomRotation(30),  # Apply random rotation
    # transforms.RandomAffine(0, translate=(0.1, 0.1)),  # Apply random affine transformations
    transforms.ToTensor()  # Convert to tensor
])

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Step 1: Training on MNIST
print("Loading MNIST dataset for pretraining...")
mnist_transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=mnist_transform)
mnist_loader = DataLoader(mnist_train, batch_size=32, shuffle=True)

# Instantiate the VAE_CINN model and move it to the device (GPU if available)
vae_cinn = VAE_CINN(
    in_channels=1, 
    latent_dim=32, 
    condition_dim=10, 
    num_control_points=10, 
    degree=3, 
    pretrained_encoder_path=None,  # Add paths if needed
    pretrained_decoder_path=None
).to(device)

# Optimizer for MNIST pretraining
optimizer = torch.optim.Adam(vae_cinn.parameters(), lr=0.001)
num_epochs = 5  # Set the number of epochs for MNIST training

print("Training on MNIST dataset...")
for epoch in range(num_epochs):
    for batch_idx, (data, _) in enumerate(mnist_loader):
        # Move data to the same device as the model (GPU or CPU)
        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon, mu, log_var = vae_cinn(data)
        loss = nn.MSELoss()(recon, data)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"MNIST Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

# Step 2: Fine-tuning on your custom dataset
print("Loading custom dataset for fine-tuning...")

# Loading your custom dataset
zip_file_path = '/fs/nexus-projects/Sketch_VLM_RL/amishab/sketch_datasets/extracted_images.zip'
with zipfile.ZipFile(zip_file_path, 'r') as archive:
    image_filenames = [f for f in archive.namelist() if f.endswith(('.png', '.jpg', '.jpeg'))]

# Step to randomly select 5000 images for fine-tuning
random.shuffle(image_filenames)  # Shuffle the filenames
subset_image_filenames = image_filenames[:5000]  # Select 5000 images

zip_image_dataset = ZipImageDataset(zip_file_path, subset_image_filenames, transform=preprocess)
zip_image_dataloader = DataLoader(zip_image_dataset, batch_size=8, shuffle=True, num_workers=1)


# Path for saving the zip file containing reconstructed images
zip_save_path = './reconstructed_images.zip'

# Optimizer and training loop
optimizer = torch.optim.Adam(vae_cinn.parameters(), lr=1e-5)
num_epochs = 10

# Create a list to store image bytes in memory
image_byte_list = []

# Start timing
start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # Track the time at the start of each epoch
    
    for batch_idx, (data, _) in enumerate(zip_image_dataloader):
        data = data.to(device)

        optimizer.zero_grad()

        # Forward pass
        recon, mu, log_var = vae_cinn(data)

        # Calculate loss
        loss = nn.MSELoss()(recon, data)
        loss.backward()
        optimizer.step()

        # Save reconstructed images only after the final epoch
        if epoch == num_epochs - 1:
            for i in range(recon.size(0)):  # Loop through the batch
                img_buffer = io.BytesIO()  # Create an in-memory byte buffer
                save_image(recon[i], img_buffer, format='png')  # Save image to the buffer
                img_buffer.seek(0)  # Go to the start of the buffer for reading
                image_name = f"epoch_{epoch+1}_batch_{batch_idx+1}_image_{i+1}.png"
                image_byte_list.append((image_name, img_buffer.read()))  # Append image name and its byte data

    # Print loss for the epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    # Time tracking
    epoch_end_time = time.time()
    elapsed_time = epoch_end_time - epoch_start_time
    total_time = time.time() - start_time
    remaining_time = (total_time / (epoch + 1)) * (num_epochs - (epoch + 1))
    
    print(f"Epoch {epoch + 1}/{num_epochs} completed in {elapsed_time:.2f} seconds.")
    print(f"Total elapsed time: {total_time:.2f} seconds. Estimated remaining time: {remaining_time:.2f} seconds.")

# After the last epoch, save all images to a zip file
with zipfile.ZipFile(zip_save_path, 'w') as zf:
    for image_name, image_bytes in image_byte_list:
        zf.writestr(image_name, image_bytes)  # Write image to zip file
    print(f"Reconstructed images saved to {zip_save_path}")

print("Training and fine-tuning completed!")


