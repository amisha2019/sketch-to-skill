import torch
import torch.nn as nn
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from scipy import interpolate
import numpy as np
from typing import List, Tuple

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

class VAE(nn.Module):
    def __init__(self, img_size:int,  in_channels: int, latent_dim: int, hidden_dims: List = None):
        super(VAE, self).__init__()

        self.encoder = Encoder(img_size, in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(img_size, latent_dim, in_channels, hidden_dims)

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


class VAE_CINN(nn.Module):
    def __init__(self,
                 img_size: int,
                 in_channels: int,
                 latent_dim: int,
                 condition_dim: int,
                 num_control_points: int,
                 degree: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VAE_CINN, self).__init__()

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_control_points = num_control_points
        self.degree = degree
        self.num_knots = num_control_points + degree + 1
        self.n_dim_total = num_control_points * 2

        self.encoder = Encoder(img_size, in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(img_size, latent_dim, in_channels, hidden_dims)
        self.fc_condition = nn.Linear(latent_dim, condition_dim)

        # Build CINN
        self.cinn = self.build_inn()
        self.trainable_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.fc_condition.parameters()) + list(self.cinn.parameters())

    def build_inn(self):
        cond = Ff.ConditionNode(self.condition_dim)
        nodes = [Ff.InputNode(self.n_dim_total)]

        for k in range(16):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': self.subnet, 'clamp': 1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)

    def subnet(self, ch_in, ch_out):
        return nn.Sequential(nn.Linear(ch_in, 512),
                             nn.ReLU(),
                             nn.Linear(512, ch_out))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: torch.Tensor, params: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(input)
        z = self.reparameterize(mu, log_var)
        condition = self.fc_condition(z)        
        if self.training and params is not None:
            z_cinn, log_jac_det = self.cinn(params, c=condition)
            recons = self.decoder(z)
            return recons, input, mu, log_var, z_cinn, log_jac_det
        else:
            z_cinn = torch.randn(input.shape[0], self.n_dim_total).to(input.device)
            recons = self.decoder(z)
            params = self.cinn(z_cinn, c=condition, rev=True)
            return params[0], recons
            # return recons, input, mu, log_var, params[0]

    def loss_function(self, *args, **kwargs) -> dict:
        recons, input, mu, log_var, z_cinn, log_jac_det = args
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # CINN loss (negative log-likelihood)
        cinn_loss = torch.mean(z_cinn**2) / 2 - torch.mean(log_jac_det) / self.n_dim_total

        loss = recons_loss + kld_weight * kld_loss + cinn_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach(), 'CINN_Loss': cinn_loss.detach()}

    def generate_trajectory(self, params, num_points=100):
        knots = torch.linspace(0, 1, self.num_knots).repeat(params.shape[0], 1).to(params.device)
        # Ensure boundary knots are repeated
        knots[:, :self.degree+1] = 0
        knots[:, -self.degree-1:] = 1

        control_points = params.reshape(-1, self.num_control_points, 2)
        
        trajectories = []
        for cp, k in zip(control_points, knots):
            tck = (k.cpu().numpy(), cp.t().cpu().numpy(), self.degree)
            t = np.linspace(0, 1, num_points)
            trajectory = interpolate.splev(t, tck)
            trajectories.append(np.array(trajectory).T)

        trajectories = torch.tensor(np.array(trajectories)).to(params.device)
        
        return trajectories
