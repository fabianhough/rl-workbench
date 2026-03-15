'''
Inspired by CleanRL
'''

import os
import yaml
import random

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

    def loss(self, observs, actions, target_q):
        # observs (#, *observ_space)
        # actions (#, )
        # target_q (#, )

        # Generating q_vals
        q_vals = self.policy(observs)

        # Gathering q values associated with actions
        q_vals = q_vals.gather(dim=1, index=actions.unsqueeze(1)).squeeze()

        # Generating the loss
        loss = F.mse_loss(input=q_vals, target=target_q)
        return loss



def rl_dqn(
    lr,
    gamma,
    epsilon,
    epsilon_decay,
    epsilon_end,
    buffer_len,
    batch_size,
    target_update,
    num_episodes,
    log_episode,
    env_str,
    env_test_seed,
    hidden_dims,
    model_name
):

    ### DQN

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

        global_steps = 0
        for episode in range(num_episodes):
            print(f'Episode {episode}', end='\r')

            # Reset Env
            observ, info = env.reset()
            steps = 0
            ep_reward = 0

            # Running through episode
            while True: # NOTE: Based on episode limit
                # Get Action
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        # Assumes action is a single action
                        action = policy_net.act(torch.tensor(observ, dtype=torch.float32).to(device))

                # Step through environment
                next_observ, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Add to buffer
                replay_buffer.add(observ, action, reward, next_observ, done)

                # Set up next step
                observ = next_observ
                ep_reward += reward
                steps += 1
                global_steps += 1

                if replay_buffer.length >= batch_size:
                    ## Network Update
                    optimizer.zero_grad()
                    s_observs, s_actions, s_rewards, s_next_observs, s_dones = \
                        replay_buffer.sample(batch_size=batch_size)

                    s_observs = torch.tensor(s_observs).to(device)
                    s_actions = torch.tensor(s_actions).to(device)
                    s_rewards = torch.tensor(s_rewards).to(device)
                    s_next_observs = torch.tensor(s_next_observs).to(device)
                    s_dones = torch.tensor(s_dones).to(device)

                    # Calculating loss
                    with torch.no_grad():
                        next_q_max, _ = target_net(s_next_observs).max(dim=1)
                        target_q = s_rewards + gamma * next_q_max * (1 - s_dones)

                    loss = policy_net.loss(s_observs, s_actions, target_q)
                    loss.backward()
                    optimizer.step()

                    mlflow.log_metrics({'loss': loss.item()}, step=global_steps)

                    if (episode + 1) % target_update == 0:
                        # Copying back weights to target
                        target_net.load_state_dict(policy_net.state_dict())
                
                # Checks for completion or cancellation
                if done:
                    # Log in MLFlow
                    mlflow.log_metrics(
                        {
                            'episode_reward': ep_reward,
                            'episode_steps': steps
                        },
                        step=episode
                    )
                    break


            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            with torch.no_grad():
                total_rewards, frames = evaluate_episode(
                    env=env,
                    net=policy_net,
                    env_seed=env_test_seed,
                    device=device
                )


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