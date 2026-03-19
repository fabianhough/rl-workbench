
import yaml

import torch
import mlflow
import gymnasium as gym

from src.agents import AgentReinforce
from src.learn import train, TrainFreq, SamplingType








if __name__ == '__main__':
    # Loading config
    with open('config-reinforce.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Configuring mlflow
    mlflow.set_tracking_uri(config.pop('tracking_uri'))
    mlflow.set_experiment(config.pop('experiment'))
    
    # Generating Environment
    env = gym.make(id=config.pop('env_str'), render_mode='rgb_array')

    # Generating agent
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    agent = AgentReinforce(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.n,
        hidden_dims=config.pop('hidden_dims'),
        device=device,
        gamma=config.pop('gamma'),
        lr=config.pop('lr')
    )

    # Train
    config['train_freq'] = TrainFreq(config['train_freq'])
    config['sampling'] = SamplingType(config['sampling'])
    with mlflow.start_run():
        train(agent=agent, env=env, **config)