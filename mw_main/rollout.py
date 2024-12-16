import os
import sys
import yaml
import torch
import imageio
from PIL import Image
from typing import Any, Dict

from rl.q_agent import QAgent, QAgentConfig
from env.metaworld_wrapper import PixelMetaWorld

def load_model_config(config_path):
    """
    Load configuration from a YAML file and return the 'q_agent' portion.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['q_agent']  # Ensuring to return only the relevant 'q_agent' portion if that's how it's structured




def create_agent_config(cfg):
    """
    Create a structured QAgentConfig from the given configuration dictionary.
    """
    actor_cfg = cfg['actor']  # Directly using cfg assuming it's already the 'q_agent' level
    critic_cfg = cfg['critic']

    agent_config = QAgentConfig(
        dropout=actor_cfg['dropout'],
        feature_dim=actor_cfg['feature_dim'],
        hidden_dim=actor_cfg['hidden_dim'],
        max_action_norm=actor_cfg['max_action_norm'],
        orth=actor_cfg['orth'],
        spatial_emb=actor_cfg['spatial_emb'],
        critic_drop=critic_cfg['drop'],
        critic_feature_dim=critic_cfg['feature_dim'],
        critic_hidden_dim=critic_cfg['hidden_dim'],
        critic_orth=critic_cfg['orth'],
        critic_spatial_emb=critic_cfg['spatial_emb'],
        lr=cfg['lr'],
        # More parameters as needed based on the full QAgentConfig expected parameters
    )
    return agent_config


def create_agent_from_config(cfg):
    """
    Instantiate the QAgent from a configuration dictionary.
    """
    agent_config = create_agent_config(cfg)  # Passing the already 'q_agent' scoped cfg

    model = QAgent(
        use_state=cfg['use_state'],
        obs_shape=cfg['obs_shape'],
        prop_shape=cfg['prop_shape'],
        action_dim=cfg['action_dim'],
        rl_camera=cfg['rl_camera'],
        cfg=agent_config  # Passing the constructed configuration
    )
    return model





def load_model(model_path: str, config_path: str, device: str) -> QAgent:
    """
    Load the trained model with the correct parameters.
    """
    cfg = load_model_config(config_path)
    model = create_agent_from_config(cfg)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def rollout(model: QAgent, env: PixelMetaWorld, num_episodes: int = 10):
    """
    Generate trajectories using the model in the environment and save as GIF.
    """
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        frames = []
        while not done:
            action = model.act(obs, eval_mode=True)  # Ensure model has an act method
            obs, reward, done, _ = env.step(action)
            frame = env.render(mode='rgb_array')
            frames.append(Image.fromarray(frame))
        imageio.mimsave(f'trajectory_{episode+1}.gif', frames, fps=30)

# Example usage of the script
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path ='/fs/nexus-projects/Sketch_VLM_RL/amishab/IBRL_teachermodel_demo3_gen_10/Assembly_seed2/model0.pt' 
    config_path = '/fs/nexus-projects/Sketch_VLM_RL/amishab/IBRL_teachermodel_demo3_gen_10/Assembly_seed2/cfg.yaml'  # Update to your actual config file path

    # Load the model
    model = load_model(model_path, config_path, device)

    # Assuming you have a proper environment setup
    env_config = {'task': 'Assembly', 'camera_name': 'corner2', 'width': 640, 'height': 480}
    env = PixelMetaWorld(**env_config)

    # Perform rollouts
    rollout(model, env)

        
        
