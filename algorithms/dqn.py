"""
DQN (Deep Q-Network)
=====================

Implementation of the Deep Q-Network algorithm from:
  Mnih et al., "Playing Atari with Deep Reinforcement Learning" (2013)
  Mnih et al., "Human-level control through deep reinforcement
  learning", Nature (2015)
  https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Applied here to CartPole-v1 (Gymnasium) instead of raw pixels, since
that lets the same core algorithm train in seconds/minutes on CPU
without a conv-net / frame-stacking pipeline. The four ideas that
make DQN work are all here:

  1. Q-learning with a neural network function approximator
     Q(s, a; theta), trained to minimize the Bellman error.
  2. Experience replay: transitions (s, a, r, s', done) are stored in
     a buffer and sampled in random mini-batches, breaking the
     temporal correlation between consecutive samples and reusing
     data efficiently.
  3. A separate, periodically-synced target network Q(s, a; theta^-)
     used to compute the TD target, which stabilizes training by
     keeping the bootstrap target fixed for a while instead of
     chasing a constantly-moving network.
  4. Epsilon-greedy exploration with decay, trading exploration for
     exploitation as training progresses.

Usage
-----
    pip install torch gymnasium --break-system-packages
    python dqn.py
"""

import argparse
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym


Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class QNetwork(nn.Module):
    """MLP approximating Q(s, ·): state -> Q-value for every action."""

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
        return self.net(state)


class ReplayBuffer:
    """Fixed-size cyclic buffer of transitions, sampled uniformly at random."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition_args) -> None:
        self.buffer.append(Transition(*transition_args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,
        batch_size: int = 64,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        eps_decay_steps: int = 10_000,
        target_update_freq: int = 500,
        device: str = ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_freq = target_update_freq

        # Online network: trained every step.
        self.q_net = QNetwork(state_dim, action_dim).to(device)
        # Target network: frozen copy, synced periodically. Provides a
        # stable bootstrap target so we aren't chasing a moving objective.
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.total_steps = 0

    def epsilon(self) -> float:
        """Linearly decay epsilon from eps_start to eps_end over eps_decay_steps."""
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        """Epsilon-greedy action selection."""
        eps = 0.0 if greedy else self.epsilon()
        if random.random() < eps:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return int(q_values.argmax(dim=1).item())

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> float | None:
        """Sample a mini-batch and take one gradient step on the Bellman error."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        # Q(s, a; theta) for the actions actually taken.
        q_values = self.q_net(states).gather(1, actions).squeeze(1)

        # TD target: r + gamma * max_a' Q(s', a'; theta^-) * (1 - done),
        # computed with the frozen target network and no gradient.
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(dim=1)[0]
            td_target = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, td_target)  # Huber loss

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for extra stability, as in the original DQN paper.
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()


def train(
    env_name: str = "CartPole-v1",
    num_episodes: int = 500,
    gamma: float = 0.99,
    lr: float = 1e-3,
    batch_size: int = 64,
    buffer_capacity: int = 50_000,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_steps: int = 10_000,
    target_update_freq: int = 500,
    warmup_steps: int = 1_000,
    solved_reward: float = 475.0,
    print_every: int = 20,
    seed: int = 0,
):
    env = gym.make(env_name)
    env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_dim, action_dim,
        lr=lr, gamma=gamma,
        buffer_capacity=buffer_capacity, batch_size=batch_size,
        eps_start=eps_start, eps_end=eps_end, eps_decay_steps=eps_decay_steps,
        target_update_freq=target_update_freq,
    )

    scores = []
    recent_scores = deque(maxlen=100)
    global_step = 0

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False

        while not done:
            # During an initial warmup period, act randomly to seed the
            # replay buffer with diverse transitions before training starts.
            if global_step < warmup_steps:
                action = random.randrange(action_dim)
            else:
                action = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, float(done))
            agent.update()

            state = next_state
            episode_reward += reward
            global_step += 1

        scores.append(episode_reward)
        recent_scores.append(episode_reward)
        avg_score = np.mean(recent_scores)

        if episode % print_every == 0:
            print(f"Episode {episode:4d} | reward {episode_reward:6.1f} | "
                  f"avg(last 100) {avg_score:6.1f} | eps {agent.epsilon():.3f}")

        if avg_score >= solved_reward and len(recent_scores) == 100:
            print(f"\nSolved in {episode} episodes! Average reward: {avg_score:.1f}")
            break

    env.close()
    return agent, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN on CartPole-v1")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--eps-decay-steps", type=int, default=10_000)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        gamma=args.gamma,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        eps_decay_steps=args.eps_decay_steps,
        target_update_freq=args.target_update_freq,
        seed=args.seed,
    )
