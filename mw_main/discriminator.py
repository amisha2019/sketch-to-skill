import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size):
        # input_size = 9, x, y, z, dx, dy, dz, goal_x, goal_y, goal_z
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.bce_loss = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

    def to(self, device):
        super().to(device)
        self.network.to(device)
        self.bce_loss.to(device)

    def forward(self, input):
        return self.network(input)
    
    def update(self, batch, demo_traj):
        loss = self.compute_loss(batch, demo_traj)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        metrics = {"discriminator/loss": loss.item()}
        return metrics

    def compute_loss(self, batch, demo_traj):
        policy_input = self.process_batch(batch)
        bs = policy_input.shape[0]
        expert_input = self.process_demo_traj(demo_traj, bs)

        expert_preds = self.forward(expert_input)
        policy_preds = self.forward(policy_input)
        
        expert_loss = self.bce_loss(expert_preds, torch.ones_like(expert_preds))
        policy_loss = self.bce_loss(policy_preds, torch.zeros_like(policy_preds))
        
        return expert_loss + policy_loss
    
    def process_demo_traj(self, demo_traj, bs):
        xyz = demo_traj["xyz"]              # shape: (data_size, 3)
        next_xyz = demo_traj["next_xyz"]    # shape: (data_size, 3)
        goal_pos = demo_traj["goal_pos"]    # shape: (data_size, 3)

        idx = torch.randperm(xyz.shape[0])[:bs]
        xyz = xyz[idx]
        next_xyz = next_xyz[idx]
        goal_pos = goal_pos[idx]

        d_xyz = next_xyz - xyz
        d_xyz = d_xyz / (torch.norm(d_xyz, dim=-1, keepdim=True) + 1e-8)

        expert_input = torch.cat([xyz, d_xyz, goal_pos], dim=-1)  # shape: (bs, 9)
        expert_input = expert_input.reshape(-1, 9)  # shape: (bs, 9)

        return expert_input

    def process_batch(self, batch):
        obs: dict[str, torch.Tensor] = batch.obs
        next_obs: dict[str, torch.Tensor] = batch.next_obs

        xyz = obs["prop"][:, :-1]
        next_xyz = next_obs["prop"][:, 1:]
        d_xyz = next_xyz - xyz
        d_xyz = d_xyz / (torch.norm(d_xyz, dim=-1, keepdim=True) + 1e-8)
        goal_pos = obs["state"][:, -3:]
        
        policy_input = torch.cat([xyz, d_xyz, goal_pos], dim=-1)  # shape: (bs, num_points - 1, 9)
        policy_input = policy_input.reshape(-1, 9)  # shape: (bs * (num_points - 1), 9)

        return policy_input

    def get_reward(self, batch):
        policy_input = self.process_batch(batch)
        with torch.no_grad():
            return -torch.log(1 - self(policy_input) + 1e-8)
