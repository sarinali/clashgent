"""Unified ADB bridge for emulator interaction."""

import io
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import EmulatorBridge
from ..config import Config
from ..game.actions import ActionType, GameAction
from ..game.state import Position

logger = logging.getLogger(__name__)


class ADBBridge(EmulatorBridge):
    """Unified bridge for BlueStacks/Android emulator via ADB.

    Handles both screenshot capture and action execution through a single
    ADB connection. Initializes eagerly in constructor.

    Features:
    - Auto-start BlueStacks if not running
    - Wait for device boot completion
    - Detect and update screen size in config
    - Screenshot capture via exec-out
    - Touch input (tap, swipe, drag)

    Attributes:
        adb_path: Path to the ADB executable
        device_id: Device address (e.g., "127.0.0.1:5555")
        screen_width: Detected screen width
        screen_height: Detected screen height
    """

    # Card positions in deck (normalized coordinates)
    CARD_POSITIONS = [
        (0.20, 0.93),  # Card 0 (leftmost)
        (0.40, 0.93),  # Card 1
        (0.60, 0.93),  # Card 2
        (0.80, 0.93),  # Card 3 (rightmost)
    ]

    # Arena bounds for card placement (normalized)
    ARENA_LEFT = 0.05
    ARENA_RIGHT = 0.95
    ARENA_TOP = 0.15
    ARENA_BOTTOM = 0.70

    def __init__(
        self,
        adb_path: str,
        device_id: str,
        config: Config,
        bluestacks_path: Optional[Path] = None,
        auto_start: bool = True,
        connect_timeout: float = 30.0,
        action_delay: float = 0.05,
    ):
        """Initialize ADB bridge and connect to emulator.

        Performs full initialization:
        1. Find ADB executable
        2. Start BlueStacks if needed
        3. Connect to ADB
        4. Wait for boot completion
        5. Detect screen size and update config

        Args:
            adb_path: Path to ADB executable (or "adb" if in PATH)
            device_id: Device address (e.g., "127.0.0.1:5555" for BlueStacks)
            config: Config object to update with detected screen size
            bluestacks_path: Path to BlueStacks app (required if auto_start=True)
            auto_start: Automatically start BlueStacks if not running
            connect_timeout: Timeout in seconds for connection attempts
            action_delay: Delay between compound actions (seconds)

        Raises:
            ValueError: If required parameters are missing
            FileNotFoundError: If ADB executable not found
            ConnectionError: If connection to emulator fails
        """
        if not device_id:
            raise ValueError(
                "device_id is required. Please configure it in your config "
                "(e.g., '127.0.0.1:5555' for BlueStacks)."
            )

        if auto_start and not bluestacks_path:
            raise ValueError(
                "bluestacks_path is required when auto_start=True. "
                "Please configure it in your config."
            )

        self.adb_path = self._find_adb(adb_path)
        self.device_id = device_id
        self.bluestacks_path = bluestacks_path
        self.auto_start = auto_start
        self.connect_timeout = connect_timeout
        self.action_delay = action_delay
        self._config = config

        # Initialize connection
        self._initialize()

    def _initialize(self) -> None:
        """Initialize connection to emulator.

        Raises:
            ConnectionError: If connection fails
        """
        logger.info("Initializing ADB Bridge...")

        # Step 1: Ensure BlueStacks is running
        if self.auto_start:
            if not self._is_bluestacks_running():
                if not self._start_bluestacks():
                    raise ConnectionError("Failed to start BlueStacks")
                logger.info("Waiting for BlueStacks to start...")
                time.sleep(5)

        # Step 2: Connect and wait for boot
        if not self._wait_for_device():
            raise ConnectionError(
                "Failed to connect to device. "
                "Make sure ADB is enabled in BlueStacks settings: "
                "Settings > Advanced > Android Debug Bridge (ADB)"
            )

        # Step 3: Detect screen size and update config
        self.screen_width, self.screen_height = self._detect_screen_size()
        self._config.screen.width = self.screen_width
        self._config.screen.height = self.screen_height

        logger.info(
            f"ADB Bridge initialized. Screen: {self.screen_width}x{self.screen_height}"
        )

    def _find_adb(self, adb_path: str) -> str:
        """Find ADB executable."""
        if adb_path == "adb":
            adb_in_path = shutil.which("adb")
            if adb_in_path:
                logger.debug(f"Found ADB in PATH: {adb_in_path}")
                return adb_in_path
            raise FileNotFoundError(
                "ADB not found in PATH. Please install Android SDK platform-tools "
                "or specify the full path to adb in your config."
            )

        if Path(adb_path).exists():
            return adb_path

        raise FileNotFoundError(f"ADB not found at: {adb_path}")

    def _build_adb_command(self, *args: str) -> list[str]:
        """Build ADB command with device specification."""
        cmd = [self.adb_path, "-s", self.device_id]
        cmd.extend(args)
        return cmd

    def _is_bluestacks_running(self) -> bool:
        """Check if BlueStacks is running."""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "BlueStacks.app"],
                capture_output=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _start_bluestacks(self) -> bool:
        """Start BlueStacks application."""
        if not self.bluestacks_path.exists():
            logger.error(f"BlueStacks not found at: {self.bluestacks_path}")
            return False

        if self._is_bluestacks_running():
            logger.info("BlueStacks is already running")
            return True

        logger.info("Starting BlueStacks...")
        try:
            subprocess.run(
                ["open", "-a", "BlueStacks"],
                check=True,
                timeout=10,
            )
            logger.info("BlueStacks start command sent")
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            logger.error(f"Failed to start BlueStacks: {e}")
            return False

    def _connect_adb(self) -> bool:
        """Connect to ADB device."""
        logger.debug(f"Connecting to ADB at {self.device_id}...")
        try:
            cmd = [self.adb_path, "connect", self.device_id]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = result.stdout.strip()
            logger.debug(f"ADB connect output: {output}")
            return "connected" in output.lower()
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"ADB connect failed: {e}")
            return False

    def _is_boot_completed(self) -> bool:
        """Check if device has finished booting."""
        try:
            cmd = self._build_adb_command("shell", "getprop", "sys.boot_completed")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() == "1"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _wait_for_device(self) -> bool:
        """Wait for device to be connected and fully booted."""
        start_time = time.time()

        # Phase 1: Wait for ADB connection
        while time.time() - start_time < self.connect_timeout:
            if self.is_connected():
                logger.info("ADB connected, waiting for boot to complete...")
                break
            self._connect_adb()
            time.sleep(1)
        else:
            logger.error(f"ADB connection failed after {self.connect_timeout}s")
            return False

        # Phase 2: Wait for boot to complete
        while time.time() - start_time < self.connect_timeout:
            if self._is_boot_completed():
                logger.info("Device fully booted and ready")
                return True
            time.sleep(1)

        logger.error(f"Device not fully booted after {self.connect_timeout}s")
        return False

    def _detect_screen_size(self, max_retries: int = 3) -> tuple[int, int]:
        """Detect screen size from device."""
        for attempt in range(max_retries):
            try:
                cmd = self._build_adb_command("shell", "wm", "size")
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if result.returncode != 0:
                    logger.debug(
                        f"Screen size attempt {attempt + 1} failed: {result.stderr.strip()}"
                    )
                    time.sleep(0.5)
                    continue

                output = result.stdout.strip()
                for line in output.split("\n"):
                    if "size:" in line.lower():
                        size_str = line.split(":")[-1].strip()
                        if "x" in size_str:
                            w, h = size_str.split("x")
                            return (int(w), int(h))

            except Exception as e:
                logger.warning(f"Screen size attempt {attempt + 1} error: {e}")
                time.sleep(0.5)

        logger.warning("Could not detect screen size, using default")
        return (1080, 1920)

    # Screenshot methods

    def capture(self) -> np.ndarray:
        """Capture screenshot via ADB screencap.

        Returns:
            RGB image as numpy array (H, W, 3)

        Raises:
            RuntimeError: If screenshot capture fails
        """
        cmd = self._build_adb_command("exec-out", "screencap", "-p")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"screencap failed: {result.stderr.decode('utf-8', errors='ignore')}"
                )

            if len(result.stdout) < 100:
                raise RuntimeError("Screenshot data too small")

            image = Image.open(io.BytesIO(result.stdout))
            return np.array(image.convert("RGB"))

        except subprocess.TimeoutExpired:
            raise RuntimeError("Screenshot capture timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to capture screenshot: {e}")

    # Action methods

    def execute(self, action: GameAction) -> bool:
        """Execute a game action.

        For PLAY_CARD actions, performs a drag from card to target.

        Args:
            action: GameAction to execute

        Returns:
            True if action executed successfully
        """
        if action.action_type == ActionType.WAIT:
            return True

        if action.action_type == ActionType.PLAY_CARD:
            return self._play_card(action.card_index, action.target_position)

        logger.warning(f"Unknown action type: {action.action_type}")
        return False

    def _play_card(
        self,
        card_index: Optional[int],
        target: Optional[Position],
    ) -> bool:
        """Execute a card play action via tap-tap (select card, then place)."""
        if card_index is None or target is None:
            logger.warning("Invalid card play: missing card_index or target")
            return False

        if not (0 <= card_index <= 3):
            logger.warning(f"Invalid card index: {card_index}")
            return False

        # Step 1: Tap the card to select it
        card_x, card_y = self.CARD_POSITIONS[card_index]
        card_px, card_py = self._normalized_to_screen(card_x, card_y)

        if not self.tap(card_px, card_py):
            logger.warning(f"Failed to tap card {card_index}")
            return False

        # Brief delay for card selection to register
        time.sleep(0.1)

        # Step 2: Tap the target position to place the card
        target_x = max(self.ARENA_LEFT, min(self.ARENA_RIGHT, target.x))
        target_y = max(self.ARENA_TOP, min(self.ARENA_BOTTOM, target.y))
        target_px, target_py = self._normalized_to_screen(target_x, target_y)

        if not self.tap(target_px, target_py):
            logger.warning(f"Failed to tap target ({target_x:.2f}, {target_y:.2f})")
            return False

        logger.debug(f"Played card {card_index} at ({target_x:.2f}, {target_y:.2f})")
        return True

    def _normalized_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert normalized coordinates to screen pixels."""
        return (
            int(x * self.screen_width),
            int(y * self.screen_height),
        )

    def tap(self, x: int, y: int) -> bool:
        """Tap at screen coordinates."""
        cmd = self._build_adb_command("shell", "input", "tap", str(x), str(y))

        try:
            result = subprocess.run(cmd, capture_output=True, timeout=5)
            if result.returncode == 0:
                logger.debug(f"Tap at ({x}, {y})")
                return True
            logger.warning(f"Tap failed: {result.stderr.decode('utf-8', errors='ignore')}")
            return False
        except subprocess.TimeoutExpired:
            logger.error("Tap command timed out")
            return False
        except Exception as e:
            logger.error(f"Tap failed: {e}")
            return False

    def tap_normalized(self, x: float, y: float) -> bool:
        """Tap at normalized coordinates (0.0 to 1.0)."""
        px, py = self._normalized_to_screen(x, y)
        return self.tap(px, py)

    def tap_card(self, card_index: int) -> bool:
        """Tap on a card in the deck."""
        if not (0 <= card_index <= 3):
            logger.warning(f"Invalid card index: {card_index}")
            return False
        card_x, card_y = self.CARD_POSITIONS[card_index]
        return self.tap_normalized(card_x, card_y)

    # Connection methods

    def is_connected(self) -> bool:
        """Check if ADB can communicate with device."""
        try:
            cmd = self._build_adb_command("get-state")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip() == "device"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def disconnect(self) -> None:
        """Disconnect from ADB device."""
        try:
            cmd = [self.adb_path, "disconnect", self.device_id]
            subprocess.run(cmd, capture_output=True, timeout=5)
            logger.debug(f"Disconnected from {self.device_id}")
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")
