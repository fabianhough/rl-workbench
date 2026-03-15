import torch



def evaluate_episode(env, net, env_seed=42, device='cpu'):
    ## Evaluate policy
    observ, info = env.reset(seed=env_seed)
    total_rewards = []
    frames = []
    while True:
        frames.append(env.render())
        action = net.act(torch.tensor(observ, dtype=torch.float32).to(device))
        observ, reward, terminated, truncated, info = env.step(action)
        total_rewards.append(reward)
        if terminated or truncated:
            break

    return total_rewards, frames



def training_episode(env, net, ret_func, ret_kwargs, device='cpu'):
    # Per episode lists
    ep_observs = []
    ep_actions = []
    ep_rewards = []

    # Reset Env
    observ, info = env.reset()

    # Running through episode
    while True: # NOTE: Based on episode limit
        # Save observ
        ep_observs.append(observ.copy().tolist()) # NOTE: Fixes list[np] warning

        # Get Action
        # Assumes action is a single action
        action = net.act(torch.tensor(observ, dtype=torch.float32).to(device))
        ep_actions.append(action)

        # Step through environment
        observ, reward, terminated, truncated, info = env.step(action)
        ep_rewards.append(reward)
        
        # Checks for completion or cancellation
        if terminated or truncated:
            # Calculate returns
            ep_returns = ret_func(rewards=ep_rewards, **ret_kwargs)

            # Reset env
            observ, info = env.reset()
            return ep_observs, ep_actions, ep_returns