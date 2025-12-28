"""Experience replay buffers for RL training."""

from dataclasses import dataclass
from typing import Generator, Optional

import numpy as np
import torch


@dataclass
class Transition:
    """Single transition in the environment.

    Attributes:
        obs: Observation at timestep t
        action: Action taken at timestep t
        reward: Reward received after taking action
        next_obs: Observation at timestep t+1
        done: Whether episode ended after this transition
    """
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class ReplayBuffer:
    """Experience replay buffer for off-policy learning.

    Stores transitions and provides random sampling for training.
    Commonly used with DQN-style algorithms.

    Attributes:
        capacity: Maximum number of transitions to store
        obs_dim: Dimension of observation vectors
    """

    def __init__(self, capacity: int, obs_dim: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum buffer size
            obs_dim: Observation dimension
        """
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.position = 0
        self.size = 0

        # Pre-allocate arrays for efficiency
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, transition: Transition) -> None:
        """Add transition to buffer.

        Args:
            transition: Transition to add
        """
        idx = self.position

        self.obs[idx] = transition.obs
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.next_obs[idx] = transition.next_obs
        self.dones[idx] = transition.done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary of batched tensors
        """
        indices = np.random.choice(self.size, batch_size, replace=False)

        return {
            "obs": torch.from_numpy(self.obs[indices]),
            "actions": torch.from_numpy(self.actions[indices]),
            "rewards": torch.from_numpy(self.rewards[indices]),
            "next_obs": torch.from_numpy(self.next_obs[indices]),
            "dones": torch.from_numpy(self.dones[indices]),
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def is_full(self) -> bool:
        """Check if buffer is at capacity."""
        return self.size >= self.capacity


class RolloutBuffer:
    """On-policy rollout buffer for PPO/A2C.

    Stores complete rollouts for on-policy learning.
    Supports GAE (Generalized Advantage Estimation) computation.

    Attributes:
        rollout_length: Number of steps per rollout
        obs_dim: Observation dimension
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
    """

    def __init__(
        self,
        rollout_length: int,
        obs_dim: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Initialize rollout buffer.

        Args:
            rollout_length: Steps per rollout
            obs_dim: Observation dimension
            gamma: Discount factor for returns
            gae_lambda: Lambda for GAE
        """
        self.rollout_length = rollout_length
        self.obs_dim = obs_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.reset()

    def reset(self) -> None:
        """Clear buffer for new rollout."""
        self.obs: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.log_probs: list[float] = []
        self.dones: list[bool] = []

        self.advantages: Optional[np.ndarray] = None
        self.returns: Optional[np.ndarray] = None

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add step to rollout.

        Args:
            obs: Observation
            action: Action taken
            reward: Reward received
            value: Value estimate from critic
            log_prob: Log probability of action
            done: Whether episode ended
        """
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def compute_returns_and_advantages(
        self,
        last_value: float,
    ) -> None:
        """Compute GAE advantages and returns.

        Uses Generalized Advantage Estimation for lower variance
        advantage estimates.

        Args:
            last_value: Value estimate for the final state
        """
        n = len(self.rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        # Convert to numpy for computation
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # GAE computation (reversed)
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
                next_non_terminal = 1.0 - float(self.dones[-1])
            else:
                next_value = values[t + 1]
                next_non_terminal = 1.0 - dones[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            self.advantages[t] = gae
            self.returns[t] = gae + values[t]

    def get_batches(
        self,
        batch_size: int,
        shuffle: bool = True,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Yield mini-batches for training.

        Args:
            batch_size: Size of each mini-batch
            shuffle: Whether to shuffle data before batching

        Yields:
            Dictionary of batched tensors
        """
        n = len(self.obs)
        indices = np.arange(n)

        if shuffle:
            np.random.shuffle(indices)

        # Convert lists to numpy arrays
        obs_array = np.array(self.obs, dtype=np.float32)
        actions_array = np.array(self.actions, dtype=np.int64)
        log_probs_array = np.array(self.log_probs, dtype=np.float32)
        values_array = np.array(self.values, dtype=np.float32)

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_indices = indices[start:end]

            yield {
                "obs": torch.from_numpy(obs_array[batch_indices]),
                "actions": torch.from_numpy(actions_array[batch_indices]),
                "old_log_probs": torch.from_numpy(log_probs_array[batch_indices]),
                "old_values": torch.from_numpy(values_array[batch_indices]),
                "advantages": torch.from_numpy(self.advantages[batch_indices]),
                "returns": torch.from_numpy(self.returns[batch_indices]),
            }

    def __len__(self) -> int:
        """Return current rollout length."""
        return len(self.obs)

    @property
    def is_full(self) -> bool:
        """Check if rollout is complete."""
        return len(self.obs) >= self.rollout_length
