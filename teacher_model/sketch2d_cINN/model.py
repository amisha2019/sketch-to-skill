import torch
import torch.nn as nn
import FrEIA.framework as Ff
import FrEIA.modules as Fm
from scipy import interpolate
import numpy as np

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
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256)
        )
    
    def forward(self, x):
        return self.cnn(x)

def subnet(ch_in, ch_out):
    return nn.Sequential(nn.Linear(ch_in, 512),
                         nn.ReLU(),
                         nn.Linear(512, ch_out))

class BSplineTrajectory2DCINN(nn.Module):
    def __init__(self, condition_dim=256, num_control_points=10, degree=3):
        super().__init__()
        self.encoder = SketchEncoder()
        self.num_control_points = num_control_points
        self.degree = degree
        self.num_knots = num_control_points + degree + 1
        self.n_dim_total = num_control_points * 2
        self.cinn = self.build_inn(condition_dim, num_control_points, self.num_knots)
        self.trainable_parameters = [p for p in self.cinn.parameters() if p.requires_grad]
        
    def build_inn(self, condition_dim, num_control_points, num_knots):
        cond = Ff.ConditionNode(condition_dim)
        nodes = [Ff.InputNode(self.n_dim_total)]  # Control points (x,y)
        nodes.append(Ff.Node(nodes[-1], Fm.Flatten, {}))

        for k in range(16):
            nodes.append(Ff.Node(nodes[-1], Fm.PermuteRandom, {'seed': k}))
            nodes.append(Ff.Node(nodes[-1], Fm.GLOWCouplingBlock,
                                 {'subnet_constructor': subnet, 'clamp': 1.0},
                                 conditions=cond))

        return Ff.ReversibleGraphNet(nodes + [cond, Ff.OutputNode(nodes[-1])], verbose=False)
    
    def forward(self, sketch, params=None):
        condition = self.encoder(sketch)
        
        if self.training:
            z, log_jac_det = self.cinn(params, c=condition)
            return z, log_jac_det
        else:
            z = torch.randn(sketch.shape[0], self.n_dim_total).to(sketch.device)

            params = self.cinn(z, c=condition, rev=True)
            return params[0]
    
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
        
        return np.array(trajectories)
        # return torch.tensor(np.array(trajectories), dtype=torch.float32).to(params.device)

def fit_bspline_trajectory(points, num_control_points=10, degree=3):
    t = np.linspace(0, 1, len(points))
    tck, _ = interpolate.splprep([points[:, 0], points[:, 1]], s=0, k=degree)
    knots, coeffs, _ = tck
    
    # Resample control points
    u = np.linspace(0, 1, num_control_points)
    control_points = np.column_stack(interpolate.splev(u, tck))
    
    # Adjust knots
    num_knots = num_control_points + degree + 1
    new_knots = np.linspace(0, 1, num_knots)
    new_knots[:degree+1] = 0
    new_knots[-degree-1:] = 1
    
    # Combine control points and knots
    params = np.concatenate([control_points.flatten(), new_knots])
    
    return torch.tensor(params, dtype=torch.float32)