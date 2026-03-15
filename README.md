# RL Workbench

Ground-up implementations of reinforcement learning algorithms in PyTorch. No wrapper libraries — each algorithm is built from scratch for clarity, modularity, and understanding.

The repository is structured so that models, agents, and environments are composable and interchangeable.

## Algorithms

| Algorithm | Status | Environment |
|-----------|--------|-------------|
| REINFORCE | ✓ | CartPole-v1 |
| DQN | ✓ | CartPole-v1 |
| A2C | | |
| PPO | | |
| SAC | | |

## Planned

- DreamerV3

## Infrastructure

All training runs are instrumented with MLflow for experiment tracking, metric logging, and artifact storage (including rendered episode GIFs).