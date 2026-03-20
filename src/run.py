
import yaml

import torch
import mlflow
import gymnasium as gym

from .learn import train, TrainFreq, SamplingType



def experiment(config, agent_builder):
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
        agent, config = agent_builder(config, env, device)

        # Setting up training frequency and sampling type
        config['train_freq'] = TrainFreq(config['train_freq'])
        config['sampling'] = SamplingType(config['sampling'])

        # Train
        train(agent=agent, env=env, **config)
