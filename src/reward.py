


def discounted_rewards_to_go(rewards, gamma):
    # Calculate the discounted rewards-to-go
    # Generates: G_t = r_(t) + gamma*r_(t+1) + gamma^2*r_(t+2) + ...
    for i in range(len(rewards))[::-1]:
        rewards[i] += gamma * rewards[i+1] if i+1 < len(rewards) else 0
    return rewards # TODO: Rewrite to protect against overwriting