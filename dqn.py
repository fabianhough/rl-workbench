'''
Inspired by CleanRL
'''

import os
import yaml

import gymnasium as gym

import mlflow
from mlflow.models import infer_signature

import logging
logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
# NOTE: Specifically for mlflow export model complaints

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

from src.utils.image import save_gif
from src.episode import training_episode, evaluate_episode
from src.buffer import ReplayBuffer
from src.model import SimpleLinearPolicyNet




class DQNPolicy(SimpleLinearPolicyNet):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, activation=nn.ReLU):
        '''
        '''
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation
        )

    def policy(self, observs):
        # Generate q values
        q_vals = self.forward(observs)
        return q_vals

    def act(self, observ):
        # Sample an action
        return self.policy(observ.unsqueeze(0)).argmax().item()

    def loss(self, observs, target_q):
        # Generating q_vals
        q_vals = self.policy(observs)

        # Generating the loss
        loss = F.mse_loss(input=q_vals, target=target_q)
        return loss



def rl_dqn(
    lr,
    gamma,
    epsilon,
    epsilon_decay,
    buffer_len,
    num_episodes,
    log_episode,
    env_str,
    env_test_seed,
    hidden_dims,
    model_name
):

    ### REINFORCE/VPG

    ## Setup

    # Creating environment and rendering for gifs in MLFlow
    env = gym.make(env_str, render_mode="rgb_array")

    # Creating policy
    input_dim = env.observation_space.shape[0] # NOTE: Only for flat states
    output_dim = env.action_space.n # NOTE: Only for discrete actions
    policy_net = DQNPolicy(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims
    )
    target_net = DQNPolicy(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims
    )
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.to(device)
    target_net.to(device)

    optimizer = Adam(
        params=policy_net.parameters(),
        lr=lr
    )
    replay_buffer = ReplayBuffer(
        buffer_len=buffer_len,
        observ_space=env.observation_space,
        action_space=env.action_space
    )


    with mlflow.start_run():

        mlflow.log_params({
            'env': env_str,
            'lr': lr,
            'gamma': gamma,
            'epsilon': epsilon,
            'epsilon_decay': epsilon_decay,
            'buffer_len': buffer_len,
            'num_episodes': num_episodes,
            'device': device
        })

        ## Training

        for episode in range(num_episodes):
            print(f'Episode {episode}', end='\r')

            ep_observs, ep_actions, ep_returns = training_episode(
                env, policy_net, discounted_rewards_to_go,
                gamma, epsilon, epsilon_decay,
                device
            )


            ## Network Update

            optimizer.zero_grad()

            # Calculate the loss
            batch_loss = policy_net.loss(
                observs=torch.tensor(batch_observs, dtype=torch.float32).to(device),
                actions=torch.tensor(batch_actions, dtype=torch.int64).to(device),
                weights=torch.tensor(batch_returns, dtype=torch.float32).to(device)
            )

            # Batch Train
            batch_loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_rewards, frames = evaluate_episode(
                    env=env,
                    net=policy_net,
                    env_seed=env_test_seed,
                    device=device
                )

            ## Log in MLFlow
            # mlflow.log_metrics(
            #     {
            #         'batch_loss': batch_loss.item(),
            #         'batch_len_avg': len(batch_observs)/episodes_per_epoch,
            #         'eval_len': len(total_rewards),
            #         'eval_reward': sum(total_rewards)
            #     },
            #     step=epoch
            # )

            if episode > 0 and (episode+1) % log_episode == 0:
                # Saving gif of performance
                eval_gif_fn = f'eval_episode_{episode:03d}.gif'
                save_gif(frames=frames, path=eval_gif_fn)
                mlflow.log_artifact(eval_gif_fn, artifact_path=f'eval')
                os.remove(eval_gif_fn)

                # Logging model
                input_sample = torch.zeros(1, input_dim).to(device)
                output_sample = policy_net.forward(input_sample).detach().cpu().numpy()
                input_sample = input_sample.cpu().numpy()
                signature = infer_signature(
                    model_input=input_sample,
                    model_output=output_sample
                )
                policy_net.cpu()
                model_info = mlflow.pytorch.log_model(
                    pytorch_model=policy_net,
                    name=model_name,
                    input_example=input_sample,
                    signature=signature
                )
                policy_net.to(device)
                mlflow.register_model(model_info.model_uri, model_name)

            

if __name__ == '__main__':

    with open('dqn-config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config.pop('tracking_uri'))
    mlflow.set_experiment(config.pop('experiment'))
    rl_dqn(**config)