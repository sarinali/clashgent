"""Abstract base class for emulator bridge."""

from abc import ABC, abstractmethod

import numpy as np

from ..game.actions import GameAction


class EmulatorBridge(ABC):
    """Unified interface for emulator interaction.

    Handles both screenshot capture and action execution.
    Implementations should manage:
    - Connection to the emulator/device
    - Screenshot capture and image conversion
    - Input execution (taps, swipes, game actions)

    Attributes:
        screen_width: Screen width in pixels
        screen_height: Screen height in pixels
    """

    screen_width: int
    screen_height: int

    # Screenshot methods

    @abstractmethod
    def capture(self) -> np.ndarray:
        """Capture current game screen.

        Returns:
            RGB image array with shape (H, W, 3) and dtype uint8.
            Values in range [0, 255].

        Raises:
            ConnectionError: If emulator is not connected
            RuntimeError: If screenshot capture fails
        """
        raise NotImplementedError

    def get_screen_size(self) -> tuple[int, int]:
        """Get screen dimensions.

        Returns:
            Tuple of (width, height) in pixels
        """
        return self.screen_width, self.screen_height

    # Action methods

    @abstractmethod
    def execute(self, action: GameAction) -> bool:
        """Execute a game action.

        Translates a high-level GameAction into emulator input.

        Args:
            action: The GameAction to execute

        Returns:
            True if action was executed successfully
        """
        raise NotImplementedError

    @abstractmethod
    def tap(self, x: int, y: int) -> bool:
        """Tap at screen coordinates.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels

        Returns:
            True if tap was successful
        """
        raise NotImplementedError

    @abstractmethod
    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> bool:
        """Perform a swipe gesture.

        Args:
            x1: Starting X coordinate
            y1: Starting Y coordinate
            x2: Ending X coordinate
            y2: Ending Y coordinate
            duration_ms: Duration of swipe in milliseconds

        Returns:
            True if swipe was successful
        """
        raise NotImplementedError

    def long_press(self, x: int, y: int, duration_ms: int = 500) -> bool:
        """Long press at screen coordinates.

        Default implementation uses swipe with same start/end point.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            duration_ms: Duration to hold in milliseconds

        Returns:
            True if long press was successful
        """
        return self.swipe(x, y, x, y, duration_ms)

    # Connection methods

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if emulator connection is active.

        Returns:
            True if connected and ready
        """
        raise NotImplementedError

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from emulator."""
        raise NotImplementedError
