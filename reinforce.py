import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical



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
        # Applying softmax on the forward pass
        return F.softmax(self._policy_params(x), dim=1)


    def act(self, state):
        '''
        Args:
            state (torch.Tensor):   A given environment single state

        Returns:
            (int, torch.Tensor):
        '''
        # Forward pass to obtain the probs of actions
        probs = self.forward(state.unsqueeze(0))

        # Generating a distribution of probs
        dist = Categorical(probs)

        # Sample an action
        action = dist.sample()

        # Return a single action and the log_prob of the action
        return action.item(), dist.log_prob(action)

