import torch
import torch.nn as nn

class SketchEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Dummy tensor to calculate the size after the CNN
        self._initialize_flattened_size()
        
        # Fully connected layer (linear layer based on calculated size)
        self.fc = nn.Linear(self.flattened_size, 256)  # Output feature vector (latent_dim)

    def _initialize_flattened_size(self):
        # Pass a dummy tensor of the same shape as the input to calculate the flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 64)  # Assuming input is 64x64 grayscale image
            dummy_output = self.cnn(dummy_input)
            self.flattened_size = dummy_output.shape[1]  # Get the flattened size dynamically

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)  # Pass through the fully connected layer
        return x


class SimplifiedBSplineModel(nn.Module):
    def __init__(self, num_control_points=10):
        super().__init__()
        self.encoder = SketchEncoder()
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),  # Input size: 256 (from encoder)
            nn.ReLU(),
            nn.Linear(512, num_control_points * 2)  # Output: control points (x, y) pairs
        )
    
    def forward(self, x):
        feat1 = self.encoder(x)  # Extract features from the image
        params = self.mlp(feat1)  # Predict control points
        return params
