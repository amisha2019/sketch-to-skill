import torch
import torch.nn as nn
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from scipy import interpolate
import numpy as np
from typing import List, Tuple
from encoder import Encoder, Encoder_Pretrained
from decoder import Decoder


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
        self.n_dim_total = num_control_points * 3

        self.encoder = Encoder(img_size, in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(img_size, latent_dim, in_channels, hidden_dims)
        self.fc_condition = nn.Linear(latent_dim * 2, condition_dim)

        # Build CINN
        self.cinn = self.build_inn()
        self.trainable_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.fc_condition.parameters()) + list(self.cinn.parameters())

    def build_inn(self):
        # currently, following: https://github.com/vislearn/conditional_INNs/tree/master/mnist_minimal_example
        # TODO: maybe try more complex architecture
        # example: https://github.com/vislearn/conditional_INNs/tree/master/colorization_minimal_example
        # the condition is input to multiple coupling layers
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

    def forward(self, sketch1: torch.Tensor, sketch2: torch.Tensor, params: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu1, log_var1 = self.encoder(sketch1)
        z1 = self.reparameterize(mu1, log_var1)
        mu2, log_var2 = self.encoder(sketch2)
        z2 = self.reparameterize(mu2, log_var2)
        z = torch.cat((z1, z2), dim=1)
        condition = self.fc_condition(z)
        if self.training and params is not None:
            z_cinn, log_jac_det = self.cinn(params, c=condition)
            recons1 = self.decoder(z1)
            recons2 = self.decoder(z2)
            return recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, z_cinn, log_jac_det
        else:
            z_cinn = torch.randn(sketch1.shape[0], self.n_dim_total).to(sketch1.device)
            recons1 = self.decoder(z1)
            recons2 = self.decoder(z2)
            params = self.cinn(z_cinn, c=condition, rev=True)
            return params[0], recons1, recons2
            # return recons, input, mu, log_var, params[0]

    def loss_function(self, *args, **kwargs) -> dict:
        recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, z_cinn, log_jac_det = args
        kld_weight = kwargs['M_N']
        recons_loss1 = F.mse_loss(recons1, sketch1)
        recons_loss2 = F.mse_loss(recons2, sketch2)
        kld_loss1 = torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=1), dim=0)
        kld_loss2 = torch.mean(-0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - log_var2.exp(), dim=1), dim=0)

        # CINN loss (negative log-likelihood)
        cinn_loss = torch.mean(z_cinn**2) / 2 - torch.mean(log_jac_det) / self.n_dim_total

        # TODO: try different weights for the losses
        loss = 0.1*recons_loss1 + kld_weight * kld_loss1 + 0.1*recons_loss2 + kld_weight * kld_loss2 + cinn_loss

        # TODO: is it possible to calculate loss based on the params?
        # TODO: is it possible to generate trajectories, and calculate loss based on the trajectory?

        losses = {
            'loss': loss,
            'Reconstruction_Loss_1': recons_loss1.detach(),
            'KLD_1': kld_loss1.detach(),
            'Reconstruction_Loss_2': recons_loss2.detach(),
            'KLD_2': kld_loss2.detach(),
            'CINN_Loss': cinn_loss.detach()
        }

        return losses

    def generate_trajectory(self, params, num_points=100):
        knots = torch.linspace(0, 1, self.num_knots).repeat(params.shape[0], 1).to(params.device)
        # Ensure boundary knots are repeated
        knots[:, :self.degree+1] = 0
        knots[:, -self.degree-1:] = 1

        control_points = params.reshape(-1, self.num_control_points, 3)
        
        trajectories = []
        for cp, k in zip(control_points, knots):
            tck = (k.cpu().numpy(), cp.t().cpu().numpy(), self.degree)
            t = np.linspace(0, 1, num_points)
            trajectory = interpolate.splev(t, tck)
            trajectories.append(np.array(trajectory).T)

        trajectories = torch.tensor(np.array(trajectories)).to(params.device)
        
        return trajectories
