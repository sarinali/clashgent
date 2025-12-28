#!/usr/bin/env python3
"""Training entry point for Clash Royale RL agent.

This script demonstrates the complete training pipeline:
1. Initialize emulator bridges (screenshot + action)
2. Initialize vision system for game state extraction
3. Create the Gym environment
4. Initialize the policy network
5. Run PPO training

Usage:
    # With real emulator (not yet implemented)
    python train.py

    # With mock components for testing
    python train.py --mock

    # Resume from checkpoint
    python train.py --resume checkpoints/checkpoint_100.pt
"""

import argparse
import logging
from pathlib import Path

import torch

from clashgent.bridges.action import ADBActionBridge, MockActionBridge
from clashgent.bridges.screenshot import ADBScreenshotBridge, MockScreenshotBridge
from clashgent.config import Config, TrainingConfig
from clashgent.rl.environment import ClashEnv
from clashgent.rl.policy import ActorCritic
from clashgent.rl.trainer import PPOTrainer
from clashgent.verifiers.registry import VerifierRegistry
from clashgent.vision.extractor import MockObjectClassifier, StateExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_mock_environment(config: Config) -> ClashEnv:
    """Create environment with mock bridges for testing.

    Args:
        config: Configuration object

    Returns:
        ClashEnv with mock components
    """
    logger.info("Creating mock environment for testing...")

    # Mock bridges
    screenshot_bridge = MockScreenshotBridge(
        width=config.screen.width,
        height=config.screen.height,
    )
    action_bridge = MockActionBridge()

    # Mock vision
    classifier = MockObjectClassifier()
    state_extractor = StateExtractor(classifier)

    # Create environment
    env = ClashEnv(
        screenshot_bridge=screenshot_bridge,
        action_bridge=action_bridge,
        state_extractor=state_extractor,
        verifiers=[],
        frame_skip=config.training.frame_skip,
        action_delay=0.0,  # No delay for mock
        obs_dim=config.training.obs_dim,
    )

    return env


def create_real_environment(config: Config) -> ClashEnv:
    """Create environment with real ADB bridges.

    Args:
        config: Configuration object

    Returns:
        ClashEnv with ADB bridges

    Raises:
        NotImplementedError: Until bridges are fully implemented
    """
    logger.info("Creating real environment with ADB bridges...")

    # ADB bridges
    screenshot_bridge = ADBScreenshotBridge(
        adb_path=config.adb.adb_path,
        device_id=config.adb.device_id,
        bluestacks_path=config.adb.bluestacks_path,
    )
    action_bridge = ADBActionBridge(
        adb_path=config.adb.adb_path,
        device_id=config.adb.device_id,
        screen_width=config.screen.width,
        screen_height=config.screen.height,
    )

    # Check connection
    if not screenshot_bridge.is_connected():
        raise ConnectionError(
            "Cannot connect to emulator. "
            "Make sure BlueStacks is running and ADB is enabled."
        )

    # TODO: Replace with trained vision model
    # For now, use mock classifier
    classifier = MockObjectClassifier()
    state_extractor = StateExtractor(classifier)

    # Create verifiers (optional reward shaping)
    # Uncomment and implement verifiers as needed:
    # verifiers = [
    #     VerifierRegistry.create("tower_damage", weight=1.0),
    #     VerifierRegistry.create("elixir_leak", weight=0.5),
    # ]
    verifiers = []

    # Create environment
    env = ClashEnv(
        screenshot_bridge=screenshot_bridge,
        action_bridge=action_bridge,
        state_extractor=state_extractor,
        verifiers=verifiers,
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
        "--mock",
        action="store_true",
        help="Use mock environment for testing",
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
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="PyTorch device",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to JSON config file (e.g., configs/macos.json). "
        "If not provided, uses default config with command-line overrides.",
    )
    args = parser.parse_args()

    # Load configuration
    if args.config:
        config = Config.from_json(args.config)
        # Override with command-line arguments if provided
        if args.device != "cuda" or not torch.cuda.is_available():
            config.device = args.device
        if args.timesteps != 1_000_000:
            config.training.total_timesteps = args.timesteps
        if args.lr != 3e-4:
            config.training.learning_rate = args.lr
        if args.checkpoint_dir != Path("checkpoints"):
            config.training.checkpoint_dir = args.checkpoint_dir
    else:
        # Create configuration from defaults and command-line args
        config = Config(
            device=args.device,
            training=TrainingConfig(
                total_timesteps=args.timesteps,
                learning_rate=args.lr,
                checkpoint_dir=args.checkpoint_dir,
            ),
        )

    logger.info("=" * 60)
    logger.info("Clashgent - Clash Royale RL Training")
    logger.info("=" * 60)
    logger.info(f"Device: {config.device}")
    logger.info(f"Total timesteps: {config.training.total_timesteps:,}")
    logger.info(f"Checkpoint dir: {config.training.checkpoint_dir}")

    # Create environment
    if args.mock:
        env = create_mock_environment(config)
    else:
        try:
            env = create_real_environment(config)
        except (ConnectionError, NotImplementedError) as e:
            logger.error(f"Failed to create real environment: {e}")
            logger.info("Falling back to mock environment...")
            env = create_mock_environment(config)

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
