"""
REINFORCE (Monte-Carlo Policy Gradient)
========================================

A clean, from-scratch implementation of the REINFORCE algorithm
(Williams, 1992), applied to the CartPole-v1 environment using
Gymnasium and PyTorch.

Core idea
---------
REINFORCE directly optimizes a parameterized policy pi_theta(a|s) by
gradient ascent on the expected return:

    J(theta) = E_{tau ~ pi_theta} [ R(tau) ]

The policy gradient theorem gives:

    grad J(theta) = E_{tau} [ sum_t grad_theta log pi_theta(a_t|s_t) * G_t ]

where G_t is the discounted return-to-go from timestep t. We estimate
this expectation with Monte-Carlo rollouts: play a full episode, then
update the policy using the actual returns observed.

Usage
-----
    pip install torch gymnasium --break-system-packages
    python reinforce.py
"""

import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym


class PolicyNetwork(nn.Module):
    """A simple MLP that maps states -> action probabilities."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)


class REINFORCEAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-2,
        gamma: float = 0.99,
        use_baseline: bool = True,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.device = device

        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Buffers for a single episode
        self.log_probs = []
        self.rewards = []

    def select_action(self, state: np.ndarray) -> int:
        """Sample an action from the current policy, cache its log-prob."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        probs = self.policy(state_t)
        dist = Categorical(probs)
        action = dist.sample()
        self.log_probs.append(dist.log_prob(action))
        return action.item()

    def store_reward(self, reward: float) -> None:
        self.rewards.append(reward)

    def _compute_returns(self) -> torch.Tensor:
        """Compute discounted return-to-go G_t for every timestep."""
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if self.use_baseline and len(returns) > 1:
            # Subtracting the mean (and normalizing by std) is a simple
            # variance-reduction baseline that does not bias the gradient
            # estimate, since E[grad log pi * b] = 0 for any state-independent b.
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self) -> float:
        """Run one REINFORCE gradient step using the completed episode."""
        returns = self._compute_returns()

        # Policy gradient loss: -sum_t log pi(a_t|s_t) * G_t
        # (negative because optimizers minimize, and we want to *ascend* J)
        policy_loss = torch.stack(
            [-log_prob * G for log_prob, G in zip(self.log_probs, returns)]
        ).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        loss_value = policy_loss.item()

        # Clear episode buffers
        self.log_probs = []
        self.rewards = []

        return loss_value


def train(
    env_name: str = "CartPole-v1",
    num_episodes: int = 1000,
    gamma: float = 0.99,
    lr: float = 1e-2,
    use_baseline: bool = True,
    solved_reward: float = 475.0,
    print_every: int = 20,
    seed: int = 0,
):
    env = gym.make(env_name)
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = REINFORCEAgent(
        state_dim, action_dim, lr=lr, gamma=gamma, use_baseline=use_baseline
    )

    scores = []
    recent_scores = deque(maxlen=100)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_reward(reward)
            episode_reward += reward
            state = next_state

        agent.update()
        scores.append(episode_reward)
        recent_scores.append(episode_reward)
        avg_score = np.mean(recent_scores)

        if episode % print_every == 0:
            print(f"Episode {episode:4d} | reward {episode_reward:6.1f} | "
                  f"avg(last 100) {avg_score:6.1f}")

        if avg_score >= solved_reward and len(recent_scores) == 100:
            print(f"\nSolved in {episode} episodes! Average reward: {avg_score:.1f}")
            break

    env.close()
    return agent, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="REINFORCE on CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--no-baseline", action="store_true",
                         help="Disable the return-normalization baseline")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        use_baseline=not args.no_baseline,
        seed=args.seed,
    )
