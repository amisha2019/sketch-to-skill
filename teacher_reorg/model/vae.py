import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.init as init
import torch.nn.functional as F
from model.encoder import Encoder, Encoder_Pretrained
from model.decoder import Decoder


class VAE(nn.Module):
    def __init__(self, img_size:int,  in_channels: int, latent_dim: int, hidden_dims: List = None, ifPretrained: bool = False, preTrainedModel_type: str = 'densenet', preTrainedModel_layers: int = 6, freeze: bool = True):
        super(VAE, self).__init__()

        self.encoder = Encoder(img_size, in_channels, latent_dim, hidden_dims) if not ifPretrained else Encoder_Pretrained(img_size, in_channels, latent_dim, preTrainedModel_type, preTrainedModel_layers, freeze)
        self.decoder = Decoder(img_size, latent_dim, in_channels, hidden_dims)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(input)
        z = self.reparameterize(mu, log_var)
        recons = self.decoder(z)
        return recons, input, mu, log_var
    
    def loss_function(self, *args, **kwargs) -> dict:
        recons, input, mu, log_var = args
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach()}
