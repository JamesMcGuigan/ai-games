"""
A2C (Advantage Actor-Critic)
=============================

Synchronous Advantage Actor-Critic, as described in OpenAI's baselines
release: https://blog.openai.com/baselines-acktr-a2c/
(A2C is the synchronous, deterministic variant of A3C from
Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning", 2016.)

Applied here to CartPole-v1 with a single environment for clarity. The
"synchronous batch of environments" trick from the blog post (stepping
several envs in lockstep and batching their transitions) is deliberately
left out to keep this a single-threaded reference implementation.

Core idea
---------
A2C combines two ideas from REINFORCE and DQN:

  * Like REINFORCE, it learns a policy pi_theta(a|s) directly and
    updates it with the policy gradient:
        grad J(theta) = E[ grad_theta log pi_theta(a_t|s_t) * A_t ]

  * Unlike REINFORCE, instead of using the noisy Monte-Carlo return G_t
    as the learning signal, it uses the ADVANTAGE:
        A_t = G_t - V_phi(s_t)
    where V_phi(s) is a learned critic (state-value function). The
    critic is trained to regress toward G_t, and its prediction acts as
    a variance-reducing baseline for the actor -- much like DQN's
    target network stabilizes Q-learning, the critic here stabilizes
    the policy gradient by giving it a low-variance signal.

  * An entropy bonus is added to the actor's loss to discourage
    premature convergence to a deterministic policy and encourage
    continued exploration (this is the third term in the combined
    A2C loss from the paper/blog post).

Total loss (single network with two heads, weights shared up to the
final layers):

  L = L_actor + c1 * L_critic - c2 * H[pi_theta]

  L_actor  = -log pi(a_t|s_t) * A_t.detach()   (advantage is NOT
             backpropagated through the critic here)
  L_critic = (V_phi(s_t) - G_t)^2               (MSE regression)
  H        = entropy of the policy distribution (bonus, hence minus)

Usage
-----
    pip install torch gymnasium --break-system-packages
    python a2c.py
"""

import argparse
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gymnasium as gym


class ActorCritic(nn.Module):
    """Shared trunk with two heads: policy (actor) and value (critic)."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)   # -> action logits
        self.critic_head = nn.Linear(hidden_dim, 1)            # -> V(s)

    def forward(self, state: torch.Tensor):
        features = self.shared(state)
        action_logits = self.actor_head(features)
        value = self.critic_head(features).squeeze(-1)
        return action_logits, value


class A2CAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 7e-4,
        gamma: float = 0.99,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_steps: int = 5,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_steps = n_steps
        self.device = device

        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Rollout buffer, cleared after every n-step update.
        self._reset_rollout()

    def _reset_rollout(self):
        self.states, self.actions = [], []
        self.rewards, self.dones = [], []
        self.log_probs, self.values, self.entropies = [], [], []

    def select_action(self, state: np.ndarray):
        """Sample an action, caching log-prob, value estimate, and entropy."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits, value = self.model(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action).squeeze(0))
        self.values.append(value.squeeze(0))
        self.entropies.append(dist.entropy().squeeze(0))
        return action.item()

    def store_step(self, reward: float, terminated: bool) -> None:
        self.rewards.append(reward)
        self.dones.append(terminated)

    @torch.no_grad()
    def _bootstrap_value(self, next_state: np.ndarray, terminated: bool) -> float:
        """V(s_{t+n}) used to bootstrap the n-step return; 0 only on a genuine
        termination (a time-limit truncation still bootstraps from next_state)."""
        if terminated:
            return 0.0
        state_t = torch.from_numpy(next_state).float().unsqueeze(0).to(self.device)
        _, value = self.model(state_t)
        return value.item()

    def update(self, next_state: np.ndarray, terminated: bool) -> dict:
        """
        Perform one n-step A2C update.

        Computes n-step returns via backward accumulation:
            G_t = r_t + gamma * G_{t+1}          (G_n = bootstrap value)
        then forms the advantage A_t = G_t - V(s_t), and combines the
        actor, critic, and entropy losses into a single backward pass.
        """
        bootstrap = self._bootstrap_value(next_state, terminated)

        returns = []
        G = bootstrap
        for r, d in zip(reversed(self.rewards), reversed(self.dones)):
            G = r + self.gamma * G * (1.0 - float(d))
            returns.insert(0, G)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        values = torch.stack(self.values)
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)

        # Advantage: how much better was the actual return than the
        # critic's prediction? Detached so actor gradients don't flow
        # into the critic through this term.
        advantages = returns - values

        actor_loss = -(log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()
        entropy_bonus = entropies.mean()

        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy_bonus

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        stats = {
            "loss": loss.item(),
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy": entropy_bonus.item(),
        }
        self._reset_rollout()
        return stats


def train(
    env_name: str = "CartPole-v1",
    num_episodes: int = 1000,
    n_steps: int = 5,
    gamma: float = 0.99,
    lr: float = 7e-4,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
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

    agent = A2CAgent(
        state_dim, action_dim,
        lr=lr, gamma=gamma,
        value_coef=value_coef, entropy_coef=entropy_coef,
        n_steps=n_steps,
    )

    scores = []
    recent_scores = deque(maxlen=100)

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        step_in_rollout = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Store `terminated` (not `done`) as the bootstrap mask: a
            # time-limit truncation cuts the rollout short but isn't a true
            # terminal, so the n-step return should still bootstrap next_state.
            agent.store_step(reward, terminated)
            episode_reward += reward
            step_in_rollout += 1
            state = next_state

            # Update every n_steps, or immediately when the episode ends
            # (whichever comes first) -- the defining rhythm of A2C's
            # n-step rollouts, as opposed to REINFORCE's full-episode
            # wait or DQN's single-transition updates.
            if step_in_rollout == n_steps or done:
                agent.update(next_state, terminated)
                step_in_rollout = 0

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
    parser = argparse.ArgumentParser(description="A2C on CartPole-v1")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--n-steps", type=int, default=5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        n_steps=args.n_steps,
        gamma=args.gamma,
        lr=args.lr,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
    )
