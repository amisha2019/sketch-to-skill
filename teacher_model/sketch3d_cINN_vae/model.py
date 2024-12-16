import torch
import torch.nn as nn
import torch.nn.functional as F
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from scipy import interpolate
import numpy as np
from typing import List, Tuple
from distances import ChamferDistance, FrechetDistance
import torch.nn.init as init
from torchvision.models import densenet121, resnet18

class Encoder_Pretrained(nn.Module):
    def __init__(self, img_size: int, in_channels: int, latent_dim: int, hidden_dims: int, preTrainedModel_type: str = 'densenet', preTrainedModel_layers: int = 6, freeze: bool = True):
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
        
        # # define a conv layer to app to pooled_output and maintain the size
        # self.convLayer = nn.Sequential(
        #     nn.Conv2d(pooled_output.size(1), pooled_output.size(1), kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(pooled_output.size(1)),
        #     nn.LeakyReLU()
        # )

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
        x = self.mlp(x.view(x.size(0), -1))
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return [mu, log_var]


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
    def __init__(self, img_size:int,  in_channels: int, latent_dim: int, hidden_dims: List = None, ifPretrained: bool = False, preTrainedModel_type: str = 'densenet', preTrainedModel_layers: int = 6, freeze: bool = True):
        super(VAE, self).__init__()

        self.encoder = Encoder(img_size, in_channels, latent_dim, hidden_dims) if not ifPretrained else Encoder_Pretrained(img_size, in_channels, latent_dim, hidden_dims, preTrainedModel_type, preTrainedModel_layers, freeze)
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


class VAE_MLP(nn.Module):
    # TODO: use xvaier initialization
    def __init__(self,
                 img_size: int,
                 in_channels: int,
                 latent_dim: int,
                 num_control_points: int,
                 degree: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VAE_MLP, self).__init__()

        self.latent_dim = latent_dim
        self.num_control_points = num_control_points
        self.degree = degree
        self.num_knots = num_control_points + degree + 1
        self.n_dim_total = num_control_points * 3
        self.init_knots()
        self.init_basis_matrix()
        # self.chamferDist = ChamferDistance_torch()
        # self.chamferDist = ChamferDistance()
        self.frechetDist = FrechetDistance()

        self.encoder = Encoder(img_size, in_channels, latent_dim, hidden_dims)
        self.decoder = Decoder(img_size, latent_dim, in_channels, hidden_dims)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, 1024),  # Increase the size of the first layer
            nn.ReLU(),
            nn.Linear(1024, 512),  # Add an extra layer
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.n_dim_total)  # Output for control points
        )

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

    def forward(self, sketch1: torch.Tensor, sketch2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu1, log_var1 = self.encoder(sketch1)
        z1 = self.reparameterize(mu1, log_var1)
        mu2, log_var2 = self.encoder(sketch2)
        z2 = self.reparameterize(mu2, log_var2)
        recons1 = self.decoder(z1)
        recons2 = self.decoder(z2)
        z = torch.cat((z1, z2), dim=1)
        params = self.mlp(z)
        return recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu
    
    def loss_function(self, *args, **kwargs) -> dict:
        recons1, sketch1, mu1, log_var1, recons2, sketch2, mu2, log_var2, params, params_gt, traj_gt = args
        kld_weight = kwargs['M_N']
        recons_loss1 = F.mse_loss(recons1, sketch1)
        recons_loss2 = F.mse_loss(recons2, sketch2)
        kld_loss1 = torch.mean(-0.5 * torch.sum(1 + log_var1 - mu1 ** 2 - log_var1.exp(), dim=1), dim=0)
        kld_loss2 = torch.mean(-0.5 * torch.sum(1 + log_var2 - mu2 ** 2 - log_var2.exp(), dim=1), dim=0)
        mse_loss = F.mse_loss(params, params_gt)

        generated_traj = self.bspline_curve(params)
        # dist_bidirectional = self.chamferDist(generated_traj, traj_gt, bidirectional=True)
        # chamfer_result = self.chamferDist(generated_traj, traj_gt)
        # frechet_result = self.frechetDist(generated_traj, traj_gt).mean()
        # breakpoint()
        traj_dist = torch.sum(torch.norm(generated_traj - traj_gt, dim=-1), dim=-1).mean()        

        # TODO: try differnt weights for the mse loss of the params
        loss = recons_loss1 + kld_weight * kld_loss1 + recons_loss2 + kld_weight * kld_loss2 + mse_loss + traj_dist
        
        return {'loss': loss, 'Reconstruction_Loss_1': recons_loss1.detach(), 'KLD_1': kld_loss1.detach(), 'Reconstruction_Loss_2': recons_loss2.detach(), 
                'KLD_2': kld_loss2.detach(), 'Param_MSE_Loss': mse_loss.detach(), 'Euclidean_Distance': traj_dist.detach()}
    

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
