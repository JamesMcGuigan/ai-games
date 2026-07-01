"""
PPO (Proximal Policy Optimization)
====================================

Implementation of PPO-Clip, from:
  Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
  https://arxiv.org/abs/1707.06347

Applied to CartPole-v1 with the same ActorCritic architecture used in
the A2C implementation -- PPO is best understood as "A2C, but safer":
it collects a longer rollout, then reuses that data for several epochs
of minibatch gradient updates, while a clipped objective keeps each
update from moving the policy too far from the one that generated the
data.

Core ideas
----------
1. GENERALIZED ADVANTAGE ESTIMATION (GAE)
   Instead of a raw n-step advantage, PPO typically uses GAE(lambda),
   an exponentially-weighted average of n-step advantages that trades
   off bias and variance:

       delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
       A_t     = delta_t + (gamma * lambda) * A_{t+1} * (1 - done_t)

   lambda=0 reduces to the 1-step TD advantage (low variance, high
   bias); lambda=1 recovers the full Monte-Carlo advantage used
   implicitly by REINFORCE (high variance, low bias).

2. CLIPPED SURROGATE OBJECTIVE
   Because the same rollout is reused for multiple gradient epochs,
   the policy being updated (pi_theta) drifts away from the policy
   that generated the data (pi_theta_old). PPO forms a probability
   ratio r_t(theta) = pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t) and
   clips it so a single update can't move the policy too far in one
   step, which is what makes PPO much more stable than vanilla A2C:

       L_clip = E[ min( r_t(theta) * A_t,
                         clip(r_t(theta), 1-eps, 1+eps) * A_t ) ]

   This is a pessimistic (lower) bound on the true objective: if the
   ratio would move too far AND that move helps, the clip caps the
   benefit; if it would move too far and HURT, the unclipped term
   still lets the gradient correct it. That asymmetry is what keeps
   PPO both safe and able to fix its mistakes.

3. MULTIPLE EPOCHS OVER MINIBATCHES
   Unlike A2C's single gradient step per rollout, PPO takes several
   epochs of minibatch SGD over the same rollout buffer, greatly
   improving sample efficiency -- the clipping is precisely what
   makes this repeated reuse safe.

Usage
-----
    pip install torch gymnasium --break-system-packages
    python ppo.py
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


class RolloutBuffer:
    """Fixed-length buffer holding one on-policy rollout of length `size`."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.states, self.actions = [], []
        self.rewards, self.dones = [], []
        self.log_probs, self.values = [], []

    def add(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def __len__(self):
        return len(self.states)


class PPOAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

        self.model = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = RolloutBuffer()

    @torch.no_grad()
    def select_action(self, state: np.ndarray):
        """Sample a_t ~ pi_theta_old(.|s_t); also return log-prob and V(s_t)
        under the CURRENT (soon to be "old") network, both cached for the
        update phase so we don't need to keep the old network around."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        logits, value = self.model(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def store_step(self, state, action, reward, done, log_prob, value):
        self.buffer.add(state, action, reward, done, log_prob, value)

    @torch.no_grad()
    def _compute_gae(self, last_value: float):
        """Generalized Advantage Estimation, computed backward over the rollout."""
        rewards = self.buffer.rewards
        values = self.buffer.values + [last_value]
        dones = self.buffer.dones

        advantages = [0.0] * len(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            # dones[t] already marks whether s_{t+1} starts a fresh episode,
            # so it doubles as the mask for the final (bootstrap) step too.
            next_non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * values[t + 1] * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae

        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = advantages + torch.tensor(values[:-1], dtype=torch.float32, device=self.device)
        return advantages, returns

    def update(self, last_value: float) -> dict:
        """
        PPO-Clip update: run n_epochs of minibatch SGD over the rollout,
        using the clipped surrogate objective + value loss + entropy bonus.
        """
        advantages, returns = self._compute_gae(last_value)
        # Advantage normalization: another standard PPO variance-reduction
        # trick, applied once per rollout (not per minibatch).
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.int64, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)

        n = len(self.buffer)
        indices = np.arange(n)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                mb_idx = indices[start:start + self.batch_size]
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long, device=self.device)

                mb_states = states[mb_idx_t]
                mb_actions = actions[mb_idx_t]
                mb_old_log_probs = old_log_probs[mb_idx_t]
                mb_advantages = advantages[mb_idx_t]
                mb_returns = returns[mb_idx_t]

                logits, values = self.model(mb_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                # Probability ratio r_t(theta) = pi_theta / pi_theta_old,
                # computed in log-space for numerical stability.
                ratio = torch.exp(new_log_probs - mb_old_log_probs)

                # Clipped surrogate objective (we minimize -L_clip).
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, mb_returns)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                stats["policy_loss"] += policy_loss.item()
                stats["value_loss"] += value_loss.item()
                stats["entropy"] += entropy.item()
                stats["clip_frac"] += clip_frac
                n_updates += 1

        for k in stats:
            stats[k] /= max(n_updates, 1)

        self.buffer.reset()
        return stats


def train(
    env_name: str = "CartPole-v1",
    total_timesteps: int = 200_000,
    rollout_len: int = 2048,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    lr: float = 3e-4,
    n_epochs: int = 4,
    batch_size: int = 64,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    solved_reward: float = 475.0,
    seed: int = 0,
):
    env = gym.make(env_name)
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(
        state_dim, action_dim,
        lr=lr, gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
        value_coef=value_coef, entropy_coef=entropy_coef,
        n_epochs=n_epochs, batch_size=batch_size,
    )

    state, _ = env.reset()
    episode_reward = 0.0
    recent_scores = deque(maxlen=100)
    episode_count = 0
    update_count = 0

    timestep = 0
    while timestep < total_timesteps:
        # --- Collect a fixed-length rollout under the current policy ---
        for _ in range(rollout_len):
            action, log_prob, value = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_reward += reward   # logged return uses the raw env reward

            # GAE resets across `done` either way (a new episode follows in the
            # buffer). On a time-limit *truncation* the episode isn't truly over,
            # so fold the lost future return into the stored reward by
            # bootstrapping V(next_state); otherwise the cutoff would be treated
            # as a real terminal state and bias the value targets.
            stored_reward = reward
            if truncated and not terminated:
                with torch.no_grad():
                    next_t = torch.from_numpy(next_state).float().unsqueeze(0).to(agent.device)
                    _, next_value = agent.model(next_t)
                stored_reward = reward + agent.gamma * next_value.item()

            agent.store_step(state, action, stored_reward, done, log_prob, value)
            state = next_state
            timestep += 1

            if done:
                episode_count += 1
                recent_scores.append(episode_reward)
                episode_reward = 0.0
                state, _ = env.reset()

        # Bootstrap value for the (possibly mid-episode) final state.
        with torch.no_grad():
            last_state_t = torch.from_numpy(state).float().unsqueeze(0).to(agent.device)
            _, last_value_t = agent.model(last_state_t)
        last_value = last_value_t.item()

        # --- Update the policy on the collected rollout ---
        stats = agent.update(last_value)
        update_count += 1

        avg_score = np.mean(recent_scores) if recent_scores else 0.0
        print(f"Update {update_count:4d} | timesteps {timestep:7d} | "
              f"episodes {episode_count:5d} | avg(last 100) {avg_score:6.1f} | "
              f"policy_loss {stats['policy_loss']:.4f} | value_loss {stats['value_loss']:.4f} | "
              f"entropy {stats['entropy']:.3f} | clip_frac {stats['clip_frac']:.2f}")

        if avg_score >= solved_reward and len(recent_scores) == 100:
            print(f"\nSolved after {timestep} timesteps ({episode_count} episodes)! "
                  f"Average reward: {avg_score:.1f}")
            break

    env.close()
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO on CartPole-v1")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--rollout-len", type=int, default=2048)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        total_timesteps=args.total_timesteps,
        rollout_len=args.rollout_len,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        lr=args.lr,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
    )
