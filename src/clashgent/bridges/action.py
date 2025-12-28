"""ADB-based action bridge implementation."""

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from .base import ActionBridge
from ..game.actions import ActionType, GameAction
from ..game.state import Position

logger = logging.getLogger(__name__)


class ADBActionBridge(ActionBridge):
    """Action bridge using ADB input commands.

    Sends touch inputs to BlueStacks or Android emulator via ADB.
    Supports tap, swipe, and long press gestures.

    Features:
    - Auto-detect ADB path
    - Share connection with ADBScreenshotBridge
    - Configurable card positions and arena bounds

    Attributes:
        adb_path: Path to the ADB executable
        device_id: Device address (default: 127.0.0.1:5555 for BlueStacks)
        screen_width: Screen width for coordinate conversion
        screen_height: Screen height for coordinate conversion
    """

    # Screen regions for card selection (normalized coordinates)
    # These define where to tap to select each card in the deck
    CARD_POSITIONS = [
        (0.20, 0.93),  # Card 0 (leftmost)
        (0.40, 0.93),  # Card 1
        (0.60, 0.93),  # Card 2
        (0.80, 0.93),  # Card 3 (rightmost)
    ]

    # Arena bounds for card placement (normalized)
    ARENA_LEFT = 0.05
    ARENA_RIGHT = 0.95
    ARENA_TOP = 0.15      # Enemy side
    ARENA_BOTTOM = 0.70   # Just above card deck

    def __init__(
        self,
        adb_path: str,
        device_id: Optional[str] = None,
        screen_width: int = 1080,
        screen_height: int = 1920,
        action_delay: float = 0.05,
    ):
        """Initialize ADB action bridge.

        Args:
            adb_path: Path to ADB executable (or "adb" if in PATH)
            device_id: Device address (required, e.g., "127.0.0.1:5555" for BlueStacks)
            screen_width: Screen width in pixels
            screen_height: Screen height in pixels
            action_delay: Delay between compound actions (seconds)
            
        Raises:
            ValueError: If device_id is not provided
            FileNotFoundError: If ADB executable not found
        """
        if not device_id:
            raise ValueError(
                "device_id is required. Please configure it in your config "
                "(e.g., '127.0.0.1:5555' for BlueStacks)."
            )
        
        self.adb_path = self._find_adb(adb_path)
        self.device_id = device_id
        self._screen_width = screen_width
        self._screen_height = screen_height
        self.action_delay = action_delay

    def _find_adb(self, adb_path: str) -> str:
        """Find ADB executable.

        Args:
            adb_path: Path to ADB executable or "adb" to search in PATH

        Returns:
            Path to ADB executable

        Raises:
            FileNotFoundError: If ADB cannot be found
        """
        # If "adb" is provided, search in PATH
        if adb_path == "adb":
            adb_in_path = shutil.which("adb")
            if adb_in_path:
                return adb_in_path
            raise FileNotFoundError(
                "ADB not found in PATH. Please install Android SDK platform-tools "
                "or specify the full path to adb in your config."
            )

        # Use explicit path
        if Path(adb_path).exists():
            return adb_path
        
        raise FileNotFoundError(
            f"ADB not found at: {adb_path}. Please check your config."
        )

    def _build_adb_command(self, *args: str) -> list[str]:
        """Build ADB command with device specification.

        Args:
            *args: ADB command arguments

        Returns:
            Complete command list
        """
        cmd = [self.adb_path]
        if self.device_id:
            cmd.extend(["-s", self.device_id])
        cmd.extend(args)
        return cmd

    def _normalized_to_screen(self, x: float, y: float) -> tuple[int, int]:
        """Convert normalized coordinates to screen pixels.

        Args:
            x: Normalized x coordinate (0.0 to 1.0)
            y: Normalized y coordinate (0.0 to 1.0)

        Returns:
            Tuple of (pixel_x, pixel_y)
        """
        return (
            int(x * self._screen_width),
            int(y * self._screen_height),
        )

    def execute(self, action: GameAction) -> bool:
        """Execute a game action.

        For PLAY_CARD actions, this performs a drag gesture from
        the card position to the target position on the arena.

        Args:
            action: GameAction to execute

        Returns:
            True if action executed successfully
        """
        if action.action_type == ActionType.WAIT:
            # Do nothing, just return success
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
        """Execute a card play action.

        Performs a drag from the card in the deck to the target position.

        Args:
            card_index: Index of card to play (0-3)
            target: Target position on arena

        Returns:
            True if successful
        """
        if card_index is None or target is None:
            logger.warning("Invalid card play: missing card_index or target")
            return False

        if not (0 <= card_index <= 3):
            logger.warning(f"Invalid card index: {card_index}")
            return False

        # Get card position in deck
        card_x, card_y = self.CARD_POSITIONS[card_index]
        card_screen_x, card_screen_y = self._normalized_to_screen(card_x, card_y)

        # Clamp target position to arena bounds
        target_x = max(self.ARENA_LEFT, min(self.ARENA_RIGHT, target.x))
        target_y = max(self.ARENA_TOP, min(self.ARENA_BOTTOM, target.y))
        target_screen_x, target_screen_y = self._normalized_to_screen(target_x, target_y)

        # Perform drag from card to target
        # Using swipe with duration to simulate drag
        success = self.swipe(
            card_screen_x, card_screen_y,
            target_screen_x, target_screen_y,
            duration_ms=200,
        )

        if success:
            logger.debug(
                f"Played card {card_index} at ({target_x:.2f}, {target_y:.2f})"
            )

        return success

    def tap(self, x: int, y: int) -> bool:
        """Perform a tap at screen coordinates via ADB.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate

        Returns:
            True if tap was successful
        """
        cmd = self._build_adb_command("shell", "input", "tap", str(x), str(y))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=5,
            )

            if result.returncode == 0:
                logger.debug(f"Tap at ({x}, {y})")
                return True
            else:
                logger.warning(
                    f"Tap failed: {result.stderr.decode('utf-8', errors='ignore')}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("Tap command timed out")
            return False
        except Exception as e:
            logger.error(f"Tap failed: {e}")
            return False

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> bool:
        """Perform a swipe gesture via ADB.

        Args:
            x1: Starting X coordinate
            y1: Starting Y coordinate
            x2: Ending X coordinate
            y2: Ending Y coordinate
            duration_ms: Duration in milliseconds

        Returns:
            True if swipe was successful
        """
        cmd = self._build_adb_command(
            "shell", "input", "swipe",
            str(x1), str(y1), str(x2), str(y2), str(duration_ms),
        )

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=max(5, duration_ms / 1000 + 2),
            )

            if result.returncode == 0:
                logger.debug(f"Swipe from ({x1}, {y1}) to ({x2}, {y2})")
                return True
            else:
                logger.warning(
                    f"Swipe failed: {result.stderr.decode('utf-8', errors='ignore')}"
                )
                return False

        except subprocess.TimeoutExpired:
            logger.error("Swipe command timed out")
            return False
        except Exception as e:
            logger.error(f"Swipe failed: {e}")
            return False

    def long_press(self, x: int, y: int, duration_ms: int = 500) -> bool:
        """Perform a long press at screen coordinates.

        Implemented as a swipe to the same point with specified duration.

        Args:
            x: X pixel coordinate
            y: Y pixel coordinate
            duration_ms: Duration to hold in milliseconds

        Returns:
            True if long press was successful
        """
        return self.swipe(x, y, x, y, duration_ms)

    def is_connected(self) -> bool:
        """Check if ADB can communicate with device.

        Returns:
            True if device is connected and responding
        """
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

    def set_screen_size(self, width: int, height: int) -> None:
        """Set the screen size for coordinate conversion.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
        """
        self._screen_width = width
        self._screen_height = height
        logger.debug(f"Screen size set to {width}x{height}")

    def tap_normalized(self, x: float, y: float) -> bool:
        """Tap at normalized coordinates.

        Args:
            x: Normalized x (0.0 to 1.0)
            y: Normalized y (0.0 to 1.0)

        Returns:
            True if successful
        """
        screen_x, screen_y = self._normalized_to_screen(x, y)
        return self.tap(screen_x, screen_y)

    def tap_card(self, card_index: int) -> bool:
        """Tap on a card in the deck.

        Args:
            card_index: Card index (0-3)

        Returns:
            True if successful
        """
        if not (0 <= card_index <= 3):
            logger.warning(f"Invalid card index: {card_index}")
            return False

        card_x, card_y = self.CARD_POSITIONS[card_index]
        return self.tap_normalized(card_x, card_y)


class MockActionBridge(ActionBridge):
    """Mock action bridge for testing and development.

    Records actions instead of executing them. Useful for:
    - Unit testing without emulator
    - Verifying action sequences
    - Debugging policy outputs
    """

    def __init__(
        self,
        screen_width: int = 1080,
        screen_height: int = 1920,
    ):
        """Initialize mock bridge.

        Args:
            screen_width: Screen width for coordinate calculations
            screen_height: Screen height for coordinate calculations
        """
        self._screen_width = screen_width
        self._screen_height = screen_height
        self.action_history: list[dict] = []
        self._connected = True

    def execute(self, action: GameAction) -> bool:
        """Record action execution.

        Args:
            action: GameAction to record

        Returns:
            Always True
        """
        self.action_history.append({
            "type": "execute",
            "action_type": action.action_type.value,
            "card_index": action.card_index,
            "target_position": (
                (action.target_position.x, action.target_position.y)
                if action.target_position else None
            ),
            "timestamp": time.time(),
        })
        return True

    def tap(self, x: int, y: int) -> bool:
        """Record tap action.

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            Always True
        """
        self.action_history.append({
            "type": "tap",
            "x": x,
            "y": y,
            "timestamp": time.time(),
        })
        return True

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> bool:
        """Record swipe action.

        Args:
            x1: Starting X
            y1: Starting Y
            x2: Ending X
            y2: Ending Y
            duration_ms: Duration

        Returns:
            Always True
        """
        self.action_history.append({
            "type": "swipe",
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "duration_ms": duration_ms,
            "timestamp": time.time(),
        })
        return True

    def is_connected(self) -> bool:
        """Mock connection is configurable.

        Returns:
            Current connection state
        """
        return self._connected

    def set_connected(self, connected: bool) -> None:
        """Set mock connection state for testing.

        Args:
            connected: Whether to report as connected
        """
        self._connected = connected

    def clear_history(self) -> None:
        """Clear action history."""
        self.action_history.clear()

    def get_last_action(self) -> Optional[dict]:
        """Get most recent recorded action.

        Returns:
            Last action dict or None if no actions
        """
        return self.action_history[-1] if self.action_history else None

    def get_action_count(self) -> int:
        """Get number of recorded actions.

        Returns:
            Number of actions in history
        """
        return len(self.action_history)
