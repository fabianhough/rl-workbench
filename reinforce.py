'''
Inspired by SpinningUp and HuggingFace
'''

import os
import yaml
import mlflow
import gymnasium as gym
import torch
import torch.nn as nn

from torch.distributions import Categorical
from torch.optim import Adam
from PIL import Image



class ModelRLReinforcePolicy(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list, activation=nn.ReLU):
        '''
        '''
        super().__init__()

        # Dynamically add linear projections and adds activations in between
        layer_dims = [input_dim] + hidden_dims + [output_dim]
        ops = []
        for i in range(len(layer_dims) - 1): # Iterate two i at a time
            ops.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                ops.append(activation())

        # Applying operations to policy
        self._policy_params = nn.Sequential(*ops)


    def forward(self, x):
        return self._policy_params(x)

    def policy(self, obs):
        # Unnormalized log probabilities
        logits = self.forward(obs)

        # Includes Softmax using log-sum-exp
        return Categorical(logits=logits)

    def act(self, observ):
        # Sample an action
        return self.policy(observ).sample().item()

    def loss(self, observs, actions, weights):
        # Generating the log_probs
        log_probs = self.policy(observs).log_prob(actions)
        # Calculating the negative loss
        return -(log_probs * weights).mean()



def save_gif(frames: list, path: str, fps: int=30):
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(
        path, save_all=True, append_images=imgs[1:],
        loop=0, duration=1000//fps
    )



def run(
    disc_gamma,
    lr,
    epochs,
    batch_size,
    env_str,
    env_test_seed,
    hidden_dims,
    experiment,
    model_name
):

    ### REINFORCE/VPG

    ## Setup

    # Creating environment and rendering for gifs in MLFlow
    env = gym.make(env_str, render_mode="rgb_array")

    # Creating policy
    input_dim = env.observation_space.shape[0] # NOTE: Only for flat states
    output_dim = env.action_space.n # NOTE: Only for discrete actions
    policy_net = ModelRLReinforcePolicy(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims
    )
    optimizer = Adam(
        params=policy_net.parameters(),
        lr=lr
    )

    mlflow.set_experiment(experiment)
    with mlflow.start_run():

        mlflow.log_params({
            'env': env_str,
            'lr': lr,
            'gamma': disc_gamma,
            'epochs': epochs,
            'batch_size': batch_size,
            'device': 'cpu'
        })

        ## Training

        for e in range(epochs):
            print('Epoch', e)
            # Batch lists for multi-episode batch training
            batch_observs = []
            batch_actions = []
            batch_returns = []

            for episode in range(batch_size):
                print('Episode', episode)
                ## Play episode

                # Per episode lists
                ep_observs = []
                ep_actions = []
                ep_rewards = []

                # Reset Env
                observ, info = env.reset()

                while True:
                    # Save observ
                    ep_observs.append(observ.copy().tolist()) # NOTE: Fixes list[np] warning

                    # Get Action
                    action = policy_net.act(torch.tensor(observ, dtype=torch.float32).unsqueeze(0))
                    ep_actions.append(action)

                    # Step through environment
                    observ, reward, terminated, truncated, info = env.step(action)
                    ep_rewards.append(reward)
                    
                    # Checks for completion or cancellation
                    if terminated or truncated:
                        # Calculate the discounted rewards-to-go
                        # Generates: G_t = r_(t) + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
                        for i in range(len(ep_rewards))[::-1]:
                            ep_rewards[i] += disc_gamma * ep_rewards[i+1] if i+1 < len(ep_rewards) else 0
                        
                        # Adding all results to batch
                        batch_observs += ep_observs
                        batch_actions += ep_actions
                        batch_returns += ep_rewards

                        # Reset env
                        observ, info = env.reset()
                        break
            
            
            ## Batch Update

            optimizer.zero_grad()

            # Calculate the loss
            batch_loss = policy_net.loss(
                observs=torch.tensor(batch_observs, dtype=torch.float32),
                actions=torch.tensor(batch_actions, dtype=torch.int32),
                weights=torch.tensor(batch_returns, dtype=torch.float32)
            )

            # Batch Train
            batch_loss.backward()
            optimizer.step()

            # TODO: Log batch loss, return, and length of batch

            ## Evaluate policy
            observ, info = env.reset(seed=env_test_seed)
            total_rewards = []
            frames = []
            while True:
                frames.append(env.render())
                action = policy_net.act(torch.tensor(observ, dtype=torch.float32).unsqueeze(0))
                observ, reward, terminated, truncated, info = env.step(action)
                total_rewards.append(reward)
                if terminated or truncated:
                    break
            
            ## Log in MLFlow
            mlflow.log_metrics(
                {
                    'batch_loss': batch_loss,
                    'batch_len': len(batch_observs),
                    'eval_len': len(total_rewards),
                    'eval_reward': sum(total_rewards)
                },
                step=e
            )
            eval_gif_fn = f'eval_epoch_{e}.gif'
            save_gif(frames=frames, path=eval_gif_fn)
            mlflow.log_artifact(eval_gif_fn, artifact_path=f'eval')
            os.remove(eval_gif_fn)

            

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config.pop('tracking_uri'))
    run(**config)