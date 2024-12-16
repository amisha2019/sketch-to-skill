import torch
import torch.nn as nn
from torchvision.models import densenet121, resnet18
from typing import List


class Encoder(nn.Module):
    def __init__(self, img_size: int, in_channels: int, latent_dim: int, hidden_dims: List = None):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim

        if hidden_dims is None:
            if img_size == 64:
                hidden_dims = [32, 64, 128, 256, 512]
            elif img_size == 224:
                hidden_dims = [16, 32, 64, 128]
            else:
                raise ValueError("Invalid image size")
            
        if img_size == 64:
            kernel_size, stride, padding = 3, 2, 1
        elif img_size == 224:
            kernel_size, stride, padding = 5, 3, 1
        else:
            raise ValueError("Invalid image size")

        modules = []
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(),
            ))
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]
    

class Encoder_Pretrained(nn.Module):
    def __init__(self, img_size: int, in_channels: int, latent_dim: int, preTrainedModel_type: str = 'Densenet', preTrainedModel_layers: int = 6, freeze: bool = True):
        super(Encoder_Pretrained, self).__init__()
        
        if preTrainedModel_type == 'Densenet':
            # using weights='DEFAULT' will download the weights from the internet
            densenet = densenet121(weights='DEFAULT') 
            self.calcPretrainedFeature = nn.Sequential(*list(densenet.features.children())[:preTrainedModel_layers])
        elif preTrainedModel_type == 'Resnet':
            resnet = resnet18(weights='DEFAULT')
            self.calcPretrainedFeature = nn.Sequential(*list(resnet.children())[:preTrainedModel_layers])
        else:
            raise ValueError("Invalid model type. Choose 'Densenet' or 'Resnet'.")
        
        # Freeze the pretrained layers
        if freeze:
            for param in self.calcPretrainedFeature.parameters():
                param.requires_grad = False
        
        # Use dummy input to calculate feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, img_size, img_size)
            encoder_output = self.calcPretrainedFeature(dummy_input) # [1, 128, 8, 8] for resnet-layer6 and densenet-layer6, [1, 64, 16, 16] for densenet-layer4, resnet-layer4
            print(encoder_output.size())
            # self.avgpool = nn.AdaptiveAvgPool2d((encoder_output.size(2)//2, encoder_output.size(3)//2))
            # pooled_output = self.avgpool(encoder_output)
            pretrainedFeatureSize = encoder_output.view(1, -1).size(1)

        # define a hiddenDim array to be used in the encoder
        hidden_dims = [pretrainedFeatureSize, 512]
        # defien a thread of mlp layers based on hidden_dims
        self.mlp = nn.Sequential()
        for i in range(len(hidden_dims) - 1):
            self.mlp.add_module(
                f"layer_{i}",
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                    nn.BatchNorm1d(hidden_dims[i+1]),
                    nn.LeakyReLU(),
                )
            )

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.calcPretrainedFeature(x)
        x = self.mlp(x.reshape(x.size(0), -1))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]
    

if __name__ == "__main__":
    # Test encoder and encoder_pretrained output shapes
    # Set up test parameters
    img_size = 64
    in_channels = 3
    latent_dim = 256
    model_type = 'Resnet'  # or 'Densenet'

    # Create instances of both encoder classes
    encoder = Encoder(img_size, in_channels, latent_dim)
    encoder_pretrained = Encoder_Pretrained(img_size, in_channels, latent_dim, preTrainedModel_type="Densenet")

    # Create a dummy input
    dummy_input = torch.randn(8, in_channels, img_size, img_size)

    # Get outputs from both encoders
    encoder_output = encoder(dummy_input)
    encoder_pretrained_output = encoder_pretrained(dummy_input)

    # Check if output shapes are the same
    print(f"Encoder output shape: {encoder_output[0].shape}, {encoder_output[1].shape}")
    print(f"Encoder_Pretrained output shape: {encoder_pretrained_output[0].shape}, {encoder_pretrained_output[1].shape}")
