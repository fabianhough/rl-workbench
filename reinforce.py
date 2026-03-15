'''
Inspired by SpinningUp and HuggingFace
'''


import gymnasium as gym
import torch
import torch.nn as nn

from torch.distributions import Categorical
from torch.optim import Adam



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
        log_probs = self.policy(obs).log_prob(actions)
        # Calculating the negative loss
        return -(log_probs * weights).mean()




def run():

    # REINFORCE/VPG

    # Hyperparameters
    # NOTE: Turn into adjustable values
    disc_gamma = 0.99
    lr = 1e-3
    epochs = 20
    batch_size = 2000
    env_str = 'CartPole-v1'
    env_test_seed = 42
    hidden_dims = [32]

    # Creating environment and rendering for gifs in MLFlow
    env = gym.make(env_str, render_mode="rgb_array")

    # Creating policy
    input_dim = env.action_space.n # NOTE: Only for discrete actions
    output_dim = env.observation_space.shape[0] # NOTE: Only for flat states
    policy_net = ModelRLReinforcePolicy(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=hidden_dims
    )
    optimizer = Adam(
        params=policy_net.parameters(),
        lr=lr
    )

    ## Training

    for e in range(epochs):
        # Batch lists for multi-episode batch training
        batch_observs = []
        batch_actions = []
        batch_returns = []

        for episode in range(batch_size):
            ## Play episode

            # Per episode lists
            ep_observs = []
            ep_actions = []
            ep_rewards = []

            # Reset Env
            observ, info = env.reset()

            while True:
                # Save observ
                ep_observs.append(observ.copy())

                # Get Action
                action = policy_net.act(torch.tensor(observ, dtype=torch.float32))
                ep_actions.append(action)

                # Step through environment
                observ, reward, terminated, truncated, info = env.step(action)
                ep_rewards.append(reward)
                
                # Checks for completion or cancellation
                if terminated or truncated:
                    # Calculate the discounted rewards-to-go
                    # Generates: G_t = r_(t) + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
                    for i in range(len(ep_rewards))[::-1]:
                        ep_rewards[i] += disc_gamma * ep_rewards[i+1] if i+1 < len(rewards) else 0
                    
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


        ## Test policy
        observ, info = env.reset(seed=env_test_seed)
        while True:
            action = policy_net.act(observ)
            observ, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        ## Log in MLFlow



