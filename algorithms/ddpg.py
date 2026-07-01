"""
DDPG (Deep Deterministic Policy Gradient)
===========================================

Implementation of DDPG, from:
  Lillicrap et al., "Continuous control with deep reinforcement
  learning" (2015)
  https://arxiv.org/abs/1509.02971

Applied to Pendulum-v1, a CONTINUOUS-action environment -- the reason
DDPG exists at all. DQN's update rule needs max_a' Q(s', a'), which is
only tractable when there are finitely many actions to enumerate. DDPG
sidesteps that by learning a deterministic actor mu_theta(s) that is
*trained to approximate* argmax_a Q(s, a), so "max over actions" is
replaced by "evaluate the actor network".

DDPG is best understood as DQN + REINFORCE/A2C's actor, glued together:

  * From DQN: a replay buffer, a Q-network (critic), and TARGET
    networks for stability -- but here BOTH actor and critic get a
    target copy, updated with Polyak (soft) averaging instead of the
    periodic hard sync DQN used.
  * From the actor-critic family: an explicit policy network. But
    unlike A2C's stochastic policy trained with a log-prob * advantage
    policy gradient, DDPG's actor is DETERMINISTIC and trained with
    the deterministic policy gradient -- directly ascending the
    critic's Q-value by backpropagating through it:
        grad_theta J = E[ grad_a Q_phi(s, a)|_{a=mu_theta(s)} * grad_theta mu_theta(s) ]
    i.e. "adjust the actor's weights so the critic's Q-value for the
    action it outputs goes up" -- no sampling, no log-probs involved.
  * Since the policy is deterministic, exploration must be added
    externally (unlike A2C, where sampling from the distribution
    already explores). DDPG adds noise to the actor's action at
    collection time -- the original paper used Ornstein-Uhlenbeck
    noise for temporally-correlated exploration; this implementation
    supports both OU and simple Gaussian noise.

Usage
-----
    pip install torch gymnasium --break-system-packages
    python ddpg.py
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


class Actor(nn.Module):
    """Deterministic policy mu_theta(s) -> action, squashed to [-action_bound, action_bound]."""

    def __init__(self, state_dim: int, action_dim: int, action_bound: float, hidden_dim: int = 256):
        super().__init__()
        self.action_bound = action_bound
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),   # squash to [-1, 1] ...
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state) * self.action_bound   # ... then scale to env's action range


class Critic(nn.Module):
    """Q_phi(s, a) -> scalar. State and action are concatenated at the input."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1)).squeeze(-1)


class ReplayBuffer:
    """Same idea as DQN's buffer: decorrelate updates, reuse transitions."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, *transition_args) -> None:
        self.buffer.append(Transition(*transition_args))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class OUNoise:
    """
    Ornstein-Uhlenbeck process: temporally-correlated noise, used by the
    original DDPG paper to encourage exploration that persists in one
    direction for a while (good for momentum-driven control tasks),
    rather than i.i.d. Gaussian noise which averages out step to step.

        dx = theta * (mu - x) * dt + sigma * dW
    """

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15,
                 sigma: float = 0.2, dt: float = 1e-2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = np.full(self.action_dim, self.mu, dtype=np.float32)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) * self.dt \
             + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """Polyak averaging: target_params <- tau * source_params + (1-tau) * target_params.

    Used instead of DQN's periodic hard sync because DDPG's actor and
    critic interact tightly -- slow, continuous tracking of the target
    networks keeps both bootstrap targets much more stable than an
    abrupt jump every K steps would."""
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_bound: float,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 5e-3,
        buffer_capacity: int = 100_000,
        batch_size: int = 128,
        noise_sigma: float = 0.2,
        use_ou_noise: bool = True,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.device = device

        # Online networks (trained every step) ...
        self.actor = Actor(state_dim, action_dim, action_bound).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        # ... and their target counterparts, initialized identically and
        # then only ever moved via slow Polyak averaging.
        self.actor_target = Actor(state_dim, action_dim, action_bound).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.replay_buffer = ReplayBuffer(buffer_capacity)

        self.use_ou_noise = use_ou_noise
        self.noise_sigma = noise_sigma
        self.ou_noise = OUNoise(action_dim, sigma=noise_sigma)

    def reset_noise(self):
        self.ou_noise.reset()

    @torch.no_grad()
    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """mu_theta(s), plus exploration noise added at collection time
        (the deterministic policy itself has no built-in exploration,
        unlike A2C's stochastic policy)."""
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action = self.actor(state_t).cpu().numpy().squeeze(0)

        if explore:
            if self.use_ou_noise:
                noise = self.ou_noise.sample()
            else:
                noise = np.random.normal(0, self.noise_sigma, size=action.shape)
            action = action + noise

        return np.clip(action, -self.action_bound, self.action_bound)

    def store_transition(self, state, action, reward, next_state, done) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(batch.action), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        # --- Critic update: standard TD/Bellman regression, just like DQN,
        # except max_a' Q(s',a') is replaced by Q(s', mu_target(s')) since
        # there's no finite action set to maximize over. ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_q = self.critic_target(next_states, next_actions)
            td_target = rewards + self.gamma * next_q * (1.0 - dones)

        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, td_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Actor update: deterministic policy gradient. We want the
        # actor's chosen action to maximize the critic's Q-value, so we
        # ascend Q(s, mu(s)) w.r.t. the actor's parameters directly --
        # equivalently, minimize -Q(s, mu(s)). The critic's weights are
        # frozen for this step (we only want gradients flowing into the
        # actor, so PyTorch's autograd handles this naturally since we
        # don't step the critic optimizer here). ---
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft-update both target networks ---
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}


def train(
    env_name: str = "Pendulum-v1",
    num_episodes: int = 200,
    gamma: float = 0.99,
    tau: float = 5e-3,
    actor_lr: float = 1e-4,
    critic_lr: float = 1e-3,
    buffer_capacity: int = 100_000,
    batch_size: int = 128,
    noise_sigma: float = 0.2,
    warmup_steps: int = 1_000,
    print_every: int = 5,
    seed: int = 0,
):
    env = gym.make(env_name)
    env.reset(seed=seed)
    env.action_space.seed(seed)   # so the warmup's action_space.sample() is reproducible
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_bound = float(env.action_space.high[0])   # assumes symmetric [-bound, bound]

    agent = DDPGAgent(
        state_dim, action_dim, action_bound,
        actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, tau=tau,
        buffer_capacity=buffer_capacity, batch_size=batch_size,
        noise_sigma=noise_sigma,
    )

    scores = []
    recent_scores = deque(maxlen=20)
    global_step = 0

    for episode in range(1, num_episodes + 1):
        state, _ = env.reset()
        agent.reset_noise()
        episode_reward = 0.0
        done = False

        while not done:
            if global_step < warmup_steps:
                # Pure-random warmup fills the buffer with diverse
                # transitions before the (initially near-random) actor
                # starts driving exploration itself.
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, explore=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Pendulum-v1 only ever truncates (200-step limit), never
            # terminates, so masking on `terminated` keeps the bootstrap
            # instead of wrongly zeroing it at every episode's last transition.
            agent.store_transition(state, action, reward, next_state, float(terminated))
            agent.update()

            state = next_state
            episode_reward += reward
            global_step += 1

        scores.append(episode_reward)
        recent_scores.append(episode_reward)
        avg_score = np.mean(recent_scores)

        if episode % print_every == 0:
            print(f"Episode {episode:4d} | reward {episode_reward:8.1f} | "
                  f"avg(last 20) {avg_score:8.1f}")

    env.close()
    return agent, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DDPG on Pendulum-v1")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--noise-sigma", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train(
        num_episodes=args.episodes,
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        noise_sigma=args.noise_sigma,
        seed=args.seed,
    )
