"""Abstract base classes for emulator bridges."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from ..game.actions import GameAction


class ScreenshotBridge(ABC):
    """Interface for capturing game screenshots from the emulator.

    Implementations should handle:
    - Connection to the emulator/device
    - Screenshot capture
    - Image format conversion to numpy array

    Example implementations:
    - ADBScreenshotBridge: Uses ADB to capture from Android emulator
    - MemoryScreenshotBridge: Reads directly from emulator memory (faster)
    - MockScreenshotBridge: Returns test images for development
    """

    @abstractmethod
    def capture(self) -> np.ndarray:
        """Capture current game screen.

        Returns:
            np.ndarray: RGB image array with shape (H, W, 3) and dtype uint8.
                       Values should be in range [0, 255].

        Raises:
            ConnectionError: If emulator is not connected
            RuntimeError: If screenshot capture fails
        """
        raise NotImplementedError

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if emulator connection is active.

        Returns:
            bool: True if connected and ready to capture screenshots
        """
        raise NotImplementedError

    def get_screen_size(self) -> tuple[int, int]:
        """Get the screen dimensions.

        Returns:
            Tuple of (width, height) in pixels

        Note:
            Default implementation captures a screenshot to get dimensions.
            Override for a more efficient implementation.
        """
        screenshot = self.capture()
        return screenshot.shape[1], screenshot.shape[0]  # width, height


class ActionBridge(ABC):
    """Interface for sending actions to the emulator.

    Implementations should handle:
    - Converting GameActions to emulator-specific commands
    - Input timing and synchronization
    - Error handling for failed inputs

    Example implementations:
    - ADBActionBridge: Uses ADB input commands
    - ScrcpyActionBridge: Uses scrcpy for input
    - MockActionBridge: Logs actions for development
    """

    @abstractmethod
    def execute(self, action: GameAction) -> bool:
        """Execute a game action.

        This method translates a high-level GameAction into
        the appropriate emulator input (tap, drag, etc.)

        Args:
            action: The GameAction to execute

        Returns:
            bool: True if action was executed successfully

        Raises:
            ConnectionError: If emulator is not connected
        """
        raise NotImplementedError

    @abstractmethod
    def tap(self, x: int, y: int) -> bool:
        """Perform a tap at screen coordinates.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels

        Returns:
            bool: True if tap was successful
        """
        raise NotImplementedError

    @abstractmethod
    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300
    ) -> bool:
        """Perform a swipe gesture.

        Args:
            x1: Starting X coordinate
            y1: Starting Y coordinate
            x2: Ending X coordinate
            y2: Ending Y coordinate
            duration_ms: Duration of swipe in milliseconds

        Returns:
            bool: True if swipe was successful
        """
        raise NotImplementedError

    def long_press(self, x: int, y: int, duration_ms: int = 500) -> bool:
        """Perform a long press at screen coordinates.

        Default implementation uses swipe with same start/end point.

        Args:
            x: X coordinate in pixels
            y: Y coordinate in pixels
            duration_ms: Duration to hold in milliseconds

        Returns:
            bool: True if long press was successful
        """
        return self.swipe(x, y, x, y, duration_ms)

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if emulator connection is active.

        Returns:
            bool: True if connected and ready to send inputs
        """
        raise NotImplementedError

    def set_screen_size(self, width: int, height: int) -> None:
        """Set the screen size for coordinate conversion.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels
        """
        self._screen_width = width
        self._screen_height = height
