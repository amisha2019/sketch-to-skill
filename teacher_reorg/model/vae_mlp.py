import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
from scipy import interpolate
from typing import List, Tuple

from model.encoder import Encoder, Encoder_Pretrained
from model.decoder import Decoder


class VAE_MLP(nn.Module):
    # TODO: use xvaier initialization
    def __init__(self,
                 img_size: int,
                 in_channels: int,
                 latent_dim: int,
                 num_control_points: int,
                 degree: int,
                 hidden_dims: List = None,
                 num_sketches: int = 2,
                 use_traj_rescale: bool = False,
                 disable_vae: bool = False,
                 **kwargs) -> None:
        super(VAE_MLP, self).__init__()

        self.latent_dim = latent_dim
        self.num_control_points = num_control_points
        self.degree = degree
        self.num_knots = num_control_points + degree + 1
        self.n_dim_total = num_control_points * 3
        self.num_sketches = num_sketches
        self.use_traj_rescale = use_traj_rescale
        self.disable_vae = disable_vae

        self.init_knots()
        self.init_basis_matrix()
        
        self.encoder = Encoder(img_size, in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(img_size, latent_dim, in_channels, hidden_dims)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * self.num_sketches, 1024),  # Increase the size of the first layer
            nn.ReLU(),
            nn.Linear(1024, 512),  # Add an extra layer
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_dim_total)  # Output for control points
        )

        # self.rot_mat = nn.Parameter(torch.rand(3, 3))
        # self.trans_vec = nn.Parameter(torch.rand(1, 3))
        # self.focal_length = nn.Parameter(torch.rand(1))

        # self.mlp = nn.Sequential(
        #     nn.Linear(latent_dim * 2, 512),
        #     nn.ReLU(),
        #     nn.Linear(512, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, self.n_dim_total)
        # )

        self.trainable_parameters = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.mlp.parameters())

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def load_pretrained_vae(self, pretrained_vae_path):
        pretrained_vae_state_dict = torch.load(pretrained_vae_path)
        
        encoder_state_dict = {k[8:]: v for k, v in pretrained_vae_state_dict.items() if k.startswith('encoder.')}
        self.encoder.load_state_dict(encoder_state_dict)
        decoder_state_dict = {k[8:]: v for k, v in pretrained_vae_state_dict.items() if k.startswith('decoder.')}
        self.decoder.load_state_dict(decoder_state_dict)
        
        print("Pretrained VAE weights loaded successfully.")

    def freeze_vae(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

    def forward(self, *args) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.num_sketches == 2:
            sketch1, sketch2 = args
            mu1, log_var1 = self.encoder(sketch1)
            z1 = self.reparameterize(mu1, log_var1)
            mu2, log_var2 = self.encoder(sketch2)
            z2 = self.reparameterize(mu2, log_var2)
            recons1 = self.decoder(z1)
            recons2 = self.decoder(z2)
            z = torch.cat((z1, z2), dim=1)
            params = self.mlp(z)
            return recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params
        else:
            sketch = args[0]
            mu, log_var = self.encoder(sketch)
            z = self.reparameterize(mu, log_var)
            recons = self.decoder(z)
            params = self.mlp(z)
            return recons, sketch, mu, log_var, params
        
    
    def forward_sketch(self, sketch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(sketch)
        z = self.reparameterize(mu, log_var)
        recons = self.decoder(z)
        return recons, sketch, mu, log_var
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def loss_function_sketch(self, *args, **kwargs) -> dict:
        recons, sketch, mu, log_var = args
        kld_weight = kwargs['M_N']
        recons_loss = F.mse_loss(recons, sketch)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach()}
    
    def loss_function(self, *args, **kwargs) -> dict:
        if self.num_sketches == 2:
            recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params, params_gt, traj_gt = args
            kld_weight = kwargs['M_N']
            recons_loss1 = F.mse_loss(recons1, sketch1)
            recons_loss2 = F.mse_loss(recons2, sketch2)
            kld_loss1 = torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=1), dim=0)
            kld_loss2 = torch.mean(-0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - log_var2.exp(), dim=1), dim=0)
        else:
            recons, sketch, mu, log_var, params, params_gt, traj_gt = args
            kld_weight = kwargs['M_N']
            recons_loss = F.mse_loss(recons, sketch)
            kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
    
        mse_loss = F.mse_loss(params, params_gt)
        generated_traj = self.bspline_curve(params)
        if self.use_traj_rescale:
            # breakpoint()
            generated_traj = self.rescale_traj(generated_traj, traj_gt[:, 0, :], traj_gt[:, -1, :])
        traj_dist = torch.sum(torch.norm(generated_traj - traj_gt, dim=-1), dim=-1).mean()

        # # multiply the rot_mat with the generated_traj
        # generated_traj = torch.matmul(generated_traj, self.rot_mat) + self.trans_vec
        # intrinsic_mat = torch.tensor([[self.focal_length, 0, img_size/2], [0, self.focal_length, img_size/2], [0, 0, 1]])
        # generated_traj = torch.matmul(generated_traj, intrinsic_mat)

        # breakpoint()

        # TODO: try differnt weights for the mse loss of the params
        if self.num_sketches == 2:
            loss = mse_loss + traj_dist if self.disable_vae else recons_loss1 + kld_weight * kld_loss1 + recons_loss2 + kld_weight * kld_loss2 + mse_loss + traj_dist
            return {'loss': loss, 'Reconstruction_Loss_1': recons_loss1.detach(), 'KLD_1': kld_loss1.detach(), 'Reconstruction_Loss_2': recons_loss2.detach(), 
                'KLD_2': kld_loss2.detach(), 'Param_MSE_Loss': mse_loss.detach(), 'Euclidean_Distance': traj_dist.detach()}
        else:
            loss = mse_loss + traj_dist if self.disable_vae else recons_loss + kld_weight * kld_loss + mse_loss + traj_dist
            return {'loss': loss, 'Reconstruction_Loss': recons_loss.detach(), 'KLD': kld_loss.detach(), 'Param_MSE_Loss': mse_loss.detach(), 'Euclidean_Distance': traj_dist.detach()}
        
        
    def rescale_traj(self, traj, start, end, rescale_z=None):
        # traj: [bs, num_points, 3]
        # start: [bs, 3]
        # end: [bs, 3]
        # Calculate the current start and end points
        current_start = traj[:, 0, :]
        current_end = traj[:, -1, :]
        
        # Calculate the scaling factor
        current_range = current_end - current_start
        target_range = end - start
        scale = target_range / current_range
        
        # Apply scaling to all points except the first and last
        scaled_traj = traj.clone()
        scaled_traj = (traj - current_start.unsqueeze(1)) * scale.unsqueeze(1)
        if rescale_z is not None:
            scaled_traj[:, :, 2] *= rescale_z.unsqueeze(1).to(traj.device)
        scaled_traj = start.unsqueeze(1) + scaled_traj

        return scaled_traj

    def generate_trajectory(self, params, num_points=100):
        knots = torch.linspace(0, 1, self.num_knots).repeat(params.shape[0], 1).to(params.device)
        # Ensure boundary knots are repeated
        knots[:, :self.degree+1] = 0
        knots[:, -self.degree-1:] = 1

        control_points = params.reshape(-1, self.num_control_points, 3)
        
        trajectories = []
        for cp, k in zip(control_points, knots):
            tck = (k.detach().cpu().numpy(), cp.t().detach().cpu().numpy(), self.degree)
            # tck = (k.cpu().numpy(), cp.t().cpu().numpy(), self.degree)
            t = np.linspace(0, 1, num_points)
            trajectory = interpolate.splev(t, tck)
            trajectories.append(np.array(trajectory).T)

        trajectories = torch.tensor(np.array(trajectories)).to(params.device)
        # In the generate_trajectory function
        # print(f"Knots: {self.knots}")
        # print(f"Basis matrix: {self.basis_matrix}")

        return trajectories
    
    def bspline_basis(self, i, p, t, knots):
        """
        Compute the i-th B-spline basis function of degree p at parameter t.
        """
        if p == 0:
            return 1.0 if knots[i] <= t < knots[i+1] else 0.0
        else:
            w1 = (t - knots[i]) / (knots[i+p] - knots[i]) if knots[i+p] != knots[i] else 0.0
            w2 = (knots[i+p+1] - t) / (knots[i+p+1] - knots[i+1]) if knots[i+p+1] != knots[i+1] else 0.0
            return w1 * self.bspline_basis(i, p-1, t, knots) + w2 * self.bspline_basis(i+1, p-1, t, knots)
    
    def init_knots(self):
        self.knots = torch.linspace(0, 1, self.num_knots)
        self.knots[:self.degree+1] = 0
        self.knots[-self.degree-1:] = 1

    def init_basis_matrix(self, num_points=100):
        t_values = torch.linspace(0, 1, num_points)
        self.basis_matrix = torch.tensor([[self.bspline_basis(i, self.degree, t, self.knots) for i in range(self.num_control_points)] for t in t_values])
        self.basis_matrix[-1, -1] = 1.0  # ensure the last point is included

    def bspline_curve(self, params, num_points=100):
        if self.basis_matrix.shape[0] != num_points:
            self.init_basis_matrix(num_points)
            
        control_points = params.reshape(-1, self.num_control_points, 3)
        trajs = torch.matmul(self.basis_matrix.to(params.device), control_points)
        
        return trajs
