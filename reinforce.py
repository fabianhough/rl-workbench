'''
Inspired by SpinningUp and HuggingFace
'''

import os
import yaml
import mlflow
import gymnasium as gym
import torch
import torch.nn as nn

import logging
logging.getLogger("mlflow.pytorch").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.requirements_utils").setLevel(logging.ERROR)
# NOTE: Specifically for mlflow export model complaints

from mlflow.models import infer_signature
from torch.distributions import Categorical
from torch.optim import Adam

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

from utils.image import save_gif


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




def evaluate(env, net, env_seed=42):
    ## Evaluate policy
    observ, info = env.reset(seed=env_seed)
    total_rewards = []
    frames = []
    while True:
        frames.append(env.render())
        action = net.act(torch.tensor(observ, dtype=torch.float32).unsqueeze(0).to(device))
        observ, reward, terminated, truncated, info = env.step(action)
        total_rewards.append(reward)
        if terminated or truncated:
            break

    return total_rewards, frames




def rl_reinforce(
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
    policy_net.to(device)
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
                    action = policy_net.act(torch.tensor(observ, dtype=torch.float32).unsqueeze(0).to(device))
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
                observs=torch.tensor(batch_observs, dtype=torch.float32).to(device),
                actions=torch.tensor(batch_actions, dtype=torch.int32).to(device),
                weights=torch.tensor(batch_returns, dtype=torch.float32).to(device)
            )

            # Batch Train
            batch_loss.backward()
            optimizer.step()

            # TODO: Log batch loss, return, and length of batch

            total_rewards, frames = evaluate(
                env=env,
                net=policy_net,
                env_seed=env_test_seed
            )
            
            ## Log in MLFlow
            mlflow.log_metrics(
                {
                    'batch_loss': batch_loss.item(),
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

            input_sample = torch.zeros(1, input_dim).to(device)
            output_sample = policy_net.forward(input_sample).detach().cpu().numpy()
            input_sample = input_sample.cpu().numpy()
            signature = infer_signature(
                model_input=input_sample,
                model_output=output_sample
            )
            policy_net.cpu()
            mlflow.pytorch.log_model(
                pytorch_model=policy_net,
                name=model_name,
                input_example=input_sample,
                signature=signature
            )
            policy_net.to(device)

            

if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(config.pop('tracking_uri'))
    rl_reinforce(**config)