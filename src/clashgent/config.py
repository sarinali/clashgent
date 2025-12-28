"""Configuration dataclasses for Clashgent."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ScreenConfig:
    """Screen and coordinate configuration.

    Calibrate these values to your specific emulator setup.
    """
    width: int = 1080
    height: int = 1920

    # Arena bounds (normalized)
    arena_left: float = 0.05
    arena_right: float = 0.95
    arena_top: float = 0.15
    arena_bottom: float = 0.70

    # Card deck positions (normalized x, y)
    card_positions: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.20, 0.93),
        (0.40, 0.93),
        (0.60, 0.93),
        (0.80, 0.93),
    ])

    # Elixir bar region (x, y, width, height)
    elixir_bar_region: tuple[float, float, float, float] = (0.08, 0.80, 0.58, 0.025)


@dataclass
class ADBConfig:
    """ADB connection configuration.
    
    Users must configure these paths for their system:
    - adb_path: Path to ADB executable (or "adb" if in PATH)
    - device_id: ADB device address (e.g., "127.0.0.1:5555" for BlueStacks)
    - bluestacks_path: Path to BlueStacks application (macOS: "/Applications/BlueStacks.app")
    """
    adb_path: str = "adb"
    device_id: Optional[str] = None
    bluestacks_path: Optional[Path] = None
    scripts_dir: Optional[Path] = None


@dataclass
class VisionConfig:
    """Vision model configuration."""
    num_classes: int = 100
    input_size: tuple[int, int] = (416, 416)
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    model_path: Optional[Path] = None


@dataclass
class TrainingConfig:
    """PPO training configuration."""
    # Learning
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # PPO specific
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    # Batching
    rollout_length: int = 2048
    batch_size: int = 64
    n_epochs: int = 10

    # Network
    hidden_dim: int = 256
    obs_dim: int = 256

    # Environment
    frame_skip: int = 4
    action_delay: float = 0.1

    # Training duration
    total_timesteps: int = 1_000_000
    log_interval: int = 1
    save_interval: int = 10

    # Paths
    checkpoint_dir: Path = Path("checkpoints")
    log_dir: Path = Path("logs")


@dataclass
class Config:
    """Complete configuration for Clashgent.

    Combines all sub-configurations for easy management.
    """
    screen: ScreenConfig = field(default_factory=ScreenConfig)
    adb: ADBConfig = field(default_factory=ADBConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Device
    device: str = "cuda"
    seed: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Config instance
        """
        # Convert string paths to Path objects
        def convert_paths(obj: dict, path_keys: list[str]) -> dict:
            """Convert string paths to Path objects for specified keys."""
            result = obj.copy()
            for key in path_keys:
                if key in result and result[key] is not None:
                    result[key] = Path(result[key])
            return result

        screen_data = convert_paths(data.get("screen", {}), [])
        adb_data = convert_paths(
            data.get("adb", {}),
            ["bluestacks_path", "scripts_dir"]
        )
        vision_data = convert_paths(
            data.get("vision", {}),
            ["model_path"]
        )
        training_data = convert_paths(
            data.get("training", {}),
            ["checkpoint_dir", "log_dir"]
        )

        return cls(
            screen=ScreenConfig(**screen_data),
            adb=ADBConfig(**adb_data),
            vision=VisionConfig(**vision_data),
            training=TrainingConfig(**training_data),
            device=data.get("device", "cuda"),
            seed=data.get("seed"),
        )

    @classmethod
    def from_json(cls, config_path: Path | str) -> "Config":
        """Load config from a JSON file.

        Args:
            config_path: Path to the JSON config file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}") from e

    def to_json(self, config_path: Path | str, indent: int = 2) -> None:
        """Save config to a JSON file.

        Args:
            config_path: Path to save the JSON config file
            indent: JSON indentation (default: 2)
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(self.to_dict(), f, indent=indent)

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        from dataclasses import asdict

        def convert_paths_to_str(obj: dict) -> dict:
            """Recursively convert Path objects to strings."""
            if isinstance(obj, dict):
                return {
                    k: str(v) if isinstance(v, Path) else convert_paths_to_str(v)
                    if isinstance(v, dict) else v
                    for k, v in obj.items()
                }
            return obj

        result = asdict(self)
        return convert_paths_to_str(result)
