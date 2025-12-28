#!/usr/bin/env python3
"""Training entry point for Clash Royale RL agent.

This script demonstrates the complete training pipeline:
1. Initialize emulator bridge (ADB connection)
2. Initialize vision system for game state extraction
3. Create the Gym environment
4. Initialize the policy network
5. Run PPO training

Usage:
    python train.py --config configs/macos.json

    # Resume from checkpoint
    python train.py --config configs/macos.json --resume checkpoints/checkpoint_100.pt
"""

import argparse
import logging
from pathlib import Path

import torch

from clashgent.bridges import ADBBridge
from clashgent.config import Config, TrainingConfig
from clashgent.rl.environment import ClashEnv
from clashgent.rl.policy import ActorCritic
from clashgent.rl.trainer import PPOTrainer
from clashgent.vision.extractor import MockObjectClassifier, StateExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_environment(config: Config) -> ClashEnv:
    """Create environment with ADB bridge.

    Args:
        config: Configuration object

    Returns:
        ClashEnv connected to emulator

    Raises:
        ConnectionError: If emulator connection fails
    """
    logger.info("Creating environment with ADB bridge...")

    # Create unified bridge (initializes connection, updates config with screen size)
    bridge = ADBBridge(
        adb_path=config.adb.adb_path,
        device_id=config.adb.device_id,
        config=config,
        bluestacks_path=config.adb.bluestacks_path,
    )

    # TODO: Replace with trained vision model
    classifier = MockObjectClassifier()
    state_extractor = StateExtractor(classifier)

    # Create environment
    env = ClashEnv(
        bridge=bridge,
        state_extractor=state_extractor,
        verifiers=[],
        frame_skip=config.training.frame_skip,
        action_delay=config.training.action_delay,
        obs_dim=config.training.obs_dim,
    )

    return env


def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train Clash Royale RL agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON config file (e.g., configs/macos.json)",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device (default: cuda if available, else cpu)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps (overrides config)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="Directory for saving checkpoints (overrides config)",
    )
    args = parser.parse_args()

    # Load configuration
    config = Config.from_json(args.config)

    # Apply command-line overrides
    if args.device:
        config.device = args.device
    elif not torch.cuda.is_available() and config.device == "cuda":
        config.device = "cpu"
    if args.timesteps:
        config.training.total_timesteps = args.timesteps
    if args.lr:
        config.training.learning_rate = args.lr
    if args.checkpoint_dir:
        config.training.checkpoint_dir = args.checkpoint_dir

    logger.info("=" * 60)
    logger.info("Clashgent - Clash Royale RL Training")
    logger.info("=" * 60)
    logger.info(f"Device: {config.device}")
    logger.info(f"Total timesteps: {config.training.total_timesteps:,}")
    logger.info(f"Checkpoint dir: {config.training.checkpoint_dir}")

    # Create environment
    env = create_environment(config)

    # Create policy network
    policy = ActorCritic(
        obs_dim=config.training.obs_dim,
        action_dim=env.action_space.n,
        hidden_dim=config.training.hidden_dim,
    )
    logger.info(f"Policy network: {sum(p.numel() for p in policy.parameters()):,} parameters")

    # Create trainer
    trainer = PPOTrainer(
        env=env,
        policy=policy,
        lr=config.training.learning_rate,
        gamma=config.training.gamma,
        gae_lambda=config.training.gae_lambda,
        clip_epsilon=config.training.clip_epsilon,
        value_coef=config.training.value_coef,
        entropy_coef=config.training.entropy_coef,
        max_grad_norm=config.training.max_grad_norm,
        rollout_length=config.training.rollout_length,
        batch_size=config.training.batch_size,
        n_epochs=config.training.n_epochs,
        device=config.device,
        checkpoint_dir=config.training.checkpoint_dir,
    )

    # Resume from checkpoint if provided
    if args.resume:
        if args.resume.exists():
            trainer.load(args.resume)
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")

    # Train
    logger.info("Starting training...")
    try:
        trainer.train(
            total_timesteps=config.training.total_timesteps,
            log_interval=config.training.log_interval,
            save_interval=config.training.save_interval,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save(config.training.checkpoint_dir / "interrupted.pt")
        logger.info("Saved interrupted checkpoint")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
