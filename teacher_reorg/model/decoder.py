import torch
import torch.nn as nn
from typing import List


class Decoder(nn.Module):
    def __init__(self, img_size: int, latent_dim: int, out_channels: int, hidden_dims: List = None):
        super(Decoder, self).__init__()
        
        if hidden_dims is None:
            if img_size == 64:
                hidden_dims = [512, 256, 128, 64, 32]
            elif img_size == 224:
                hidden_dims = [128, 64, 32, 16]
            else:
                raise ValueError("Invalid image size")

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[0] * 4)
        self.hidden_dims = hidden_dims

        modules = []
        if img_size == 64:
            kernel_size, stride, padding = 3, 2, 1
            out_paddings = [1, 1, 1, 1, 1]
        elif img_size == 224:
            kernel_size, stride, padding = 5, 3, 1
            out_paddings = [2, 0, 2, 2]
        else:
            raise ValueError("Invalid image size")

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding,
                                       output_padding=out_paddings[i]),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=kernel_size,
                                               stride=stride,
                                               padding=padding,
                                               output_padding=out_paddings[-1]),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                                      kernel_size=3, padding=1),
                            nn.Tanh())

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.hidden_dims[0], 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result