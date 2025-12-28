"""PPO trainer for Clash Royale agent."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .buffer import RolloutBuffer
from .environment import ClashEnv
from .policy import ActorCritic

logger = logging.getLogger(__name__)


class PPOTrainer:
    """Proximal Policy Optimization trainer for Clash Royale.

    Implements the PPO algorithm with clipped surrogate objective
    for stable policy updates.

    Attributes:
        env: Clash Royale environment
        policy: Actor-Critic policy network
        optimizer: Network optimizer
        buffer: Rollout buffer for experience collection
    """

    def __init__(
        self,
        env: ClashEnv,
        policy: ActorCritic,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        rollout_length: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
    ):
        """Initialize PPO trainer.

        Args:
            env: Gymnasium environment
            policy: Actor-Critic network
            lr: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            rollout_length: Steps per rollout
            batch_size: Mini-batch size for updates
            n_epochs: Number of epochs per update
            device: PyTorch device
            checkpoint_dir: Directory for saving checkpoints
        """
        self.env = env
        self.policy = policy.to(device)
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.n_epochs = n_epochs

        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)

        # Buffer
        self.buffer = RolloutBuffer(
            rollout_length,
            env.observation_space.shape[0],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )

        # Tracking
        self.total_steps = 0
        self.episode_count = 0
        self.best_reward = float("-inf")

        # Create checkpoint directory
        if checkpoint_dir:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def collect_rollout(self) -> dict:
        """Collect rollout of experience from environment.

        Returns:
            Dictionary with rollout statistics
        """
        self.buffer.reset()
        obs, info = self.env.reset()

        episode_rewards = []
        current_episode_reward = 0.0

        for step in range(self.rollout_length):
            # Convert observation to tensor
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)

            # Get action from policy
            with torch.no_grad():
                action, value = self.policy.get_action(obs_tensor)
                action_logits, _ = self.policy(obs_tensor)
                dist = torch.distributions.Categorical(logits=action_logits)
                log_prob = dist.log_prob(action)

            # Execute action
            action_np = action.cpu().numpy()[0]
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated

            # Store transition
            self.buffer.push(
                obs=obs,
                action=action_np,
                reward=reward,
                value=value.cpu().numpy().item(),
                log_prob=log_prob.cpu().numpy().item(),
                done=done,
            )

            current_episode_reward += reward
            obs = next_obs
            self.total_steps += 1

            if done:
                episode_rewards.append(current_episode_reward)
                current_episode_reward = 0.0
                self.episode_count += 1
                obs, info = self.env.reset()

        # Compute value of final state for GAE
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
            last_value = self.policy.get_value(obs_tensor).cpu().numpy().item()

        self.buffer.compute_returns_and_advantages(last_value)

        return {
            "episode_rewards": episode_rewards,
            "mean_reward": np.mean(episode_rewards) if episode_rewards else 0.0,
            "num_episodes": len(episode_rewards),
        }

    def train_step(self) -> dict:
        """Perform one PPO training iteration.

        Returns:
            Dictionary with training metrics
        """
        # Collect experience
        rollout_info = self.collect_rollout()

        # PPO update
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        for epoch in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.batch_size):
                # Move batch to device
                obs = batch["obs"].to(self.device)
                actions = batch["actions"].to(self.device)
                old_log_probs = batch["old_log_probs"].to(self.device)
                advantages = batch["advantages"].to(self.device)
                returns = batch["returns"].to(self.device)

                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Get current policy outputs
                log_probs, entropy, values = self.policy.evaluate_actions(obs, actions)
                values = values.squeeze(-1)

                # Compute ratio
                ratio = torch.exp(log_probs - old_log_probs)

                # Clipped surrogate objective
                surr1 = ratio * advantages
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.clip_epsilon,
                    1.0 + self.clip_epsilon,
                ) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                value_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                num_updates += 1

        # Compute averages
        metrics = {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": total_entropy / num_updates,
            **rollout_info,
        }

        return metrics

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        save_interval: int = 10,
    ) -> None:
        """Main training loop.

        Args:
            total_timesteps: Total environment steps to train
            log_interval: Iterations between logging
            save_interval: Iterations between checkpoints
        """
        iteration = 0

        logger.info(f"Starting training for {total_timesteps} timesteps")
        logger.info(f"Rollout length: {self.rollout_length}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Epochs per update: {self.n_epochs}")

        while self.total_steps < total_timesteps:
            metrics = self.train_step()
            iteration += 1

            # Logging
            if iteration % log_interval == 0:
                logger.info(
                    f"Iter {iteration:4d} | "
                    f"Steps: {self.total_steps:7d} | "
                    f"Episodes: {self.episode_count:4d} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Policy: {metrics['policy_loss']:.4f} | "
                    f"Value: {metrics['value_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f} | "
                    f"Mean Reward: {metrics['mean_reward']:.2f}"
                )

            # Checkpointing
            if self.checkpoint_dir and iteration % save_interval == 0:
                self.save(self.checkpoint_dir / f"checkpoint_{iteration}.pt")

                # Save best model
                if metrics["mean_reward"] > self.best_reward:
                    self.best_reward = metrics["mean_reward"]
                    self.save(self.checkpoint_dir / "best_policy.pt")
                    logger.info(f"New best reward: {self.best_reward:.2f}")

        logger.info("Training complete!")
        logger.info(f"Total steps: {self.total_steps}")
        logger.info(f"Total episodes: {self.episode_count}")
        logger.info(f"Best reward: {self.best_reward:.2f}")

        # Save final model
        if self.checkpoint_dir:
            self.save(self.checkpoint_dir / "final_policy.pt")

    def save(self, path: Path) -> None:
        """Save trainer state to checkpoint.

        Args:
            path: Checkpoint file path
        """
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "best_reward": self.best_reward,
        }, path)
        logger.debug(f"Saved checkpoint to {path}")

    def load(self, path: Path) -> None:
        """Load trainer state from checkpoint.

        Args:
            path: Checkpoint file path
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_steps = checkpoint["total_steps"]
        self.episode_count = checkpoint["episode_count"]
        self.best_reward = checkpoint.get("best_reward", float("-inf"))
        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"Resuming from step {self.total_steps}, episode {self.episode_count}")
