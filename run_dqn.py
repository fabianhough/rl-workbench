
import yaml

import torch
import mlflow
import gymnasium as gym

from src.agents import AgentDQN
from src.learn import train, TrainFreq, SamplingType








if __name__ == '__main__':
    # Loading config
    with open('config-dqn.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Configuring mlflow
    mlflow.set_tracking_uri(config.pop('tracking_uri'))
    mlflow.set_experiment(config.pop('experiment'))
    mlflow.enable_system_metrics_logging()

    # Starting MLFlow
    with mlflow.start_run():

        # Log Parameters
        mlflow.log_params(params=config)
    
        # Generating Environment
        env = gym.make(id=config.pop('env_str'), render_mode='rgb_array')

        # Generating agent
        device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
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

        # Setting up training frequency and sampling type
        config['train_freq'] = TrainFreq(config['train_freq'])
        config['sampling'] = SamplingType(config['sampling'])

        # Train
        train(agent=agent, env=env, **config)
