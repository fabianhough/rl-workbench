
import yaml

import torch
import mlflow
import gymnasium as gym

from src import experiment
from src.agents import AgentDQN
from src.learn import train, TrainFreq, SamplingType



def agent_builder_dqn(config, env, device):
    agent = AgentDQN(
        input_dim=env.observation_space.shape[0],
        output_dim=env.action_space.n,
        hidden_dims=config.pop('hidden_dims'),
        device=device,
        env=env,
        gamma=config.pop('gamma'),
        lr=config.pop('lr'),
        epsilon_start=config.pop('epsilon_start'),
        epsilon_end=config.pop('epsilon_end'),
        epsilon_decay=config.pop('epsilon_decay'),
        target_update_freq=config.pop('target_update_freq')
    )
    return agent, config


if __name__ == '__main__':
    # Loading config
    with open('config-dqn.yaml', 'r') as f:
        config = yaml.safe_load(f)

    experiment(config, agent_builder_dqn)
