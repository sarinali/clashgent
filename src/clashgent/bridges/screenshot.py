"""ADB-based screenshot bridge implementation."""

import io
import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .base import ScreenshotBridge

logger = logging.getLogger(__name__)


class ADBScreenshotBridge(ScreenshotBridge):
    """Screenshot bridge using ADB (Android Debug Bridge).

    This bridge captures screenshots from BlueStacks or Android emulator
    using ADB commands. It auto-initializes the emulator if not running.

    Features:
    - Auto-detect ADB path
    - Auto-start BlueStacks if not running
    - Auto-connect to ADB
    - Fast screenshot capture via exec-out

    Attributes:
        adb_path: Path to the ADB executable
        device_id: Device address (default: 127.0.0.1:5555 for BlueStacks)
        bluestacks_path: Path to BlueStacks application
    """

    def __init__(
        self,
        adb_path: str,
        device_id: Optional[str] = None,
        bluestacks_path: Optional[Path] = None,
        auto_start: bool = True,
        connect_timeout: float = 30.0,
    ):
        """Initialize ADB screenshot bridge.

        Args:
            adb_path: Path to ADB executable (or "adb" if in PATH)
            device_id: Device address (required, e.g., "127.0.0.1:5555" for BlueStacks)
            bluestacks_path: Path to BlueStacks app (required if auto_start=True)
            auto_start: Automatically start BlueStacks if not running
            connect_timeout: Timeout in seconds for connection attempts
            
        Raises:
            ValueError: If required parameters are missing
            FileNotFoundError: If ADB executable not found
        """
        if not device_id:
            raise ValueError(
                "device_id is required. Please configure it in your config "
                "(e.g., '127.0.0.1:5555' for BlueStacks)."
            )
        
        self.adb_path = self._find_adb(adb_path)
        self.device_id = device_id
        
        if auto_start and not bluestacks_path:
            raise ValueError(
                "bluestacks_path is required when auto_start=True. "
                "Please configure it in your config "
                "(e.g., '/Applications/BlueStacks.app' on macOS)."
            )
        self.bluestacks_path = bluestacks_path
        self.auto_start = auto_start
        self.connect_timeout = connect_timeout

        self._screen_size: Optional[tuple[int, int]] = None
        self._initialized = False

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
                logger.debug(f"Found ADB in PATH: {adb_in_path}")
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

    def _is_bluestacks_running(self) -> bool:
        """Check if BlueStacks is running.

        Returns:
            True if BlueStacks process is found
        """
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
        """Start BlueStacks application.

        Returns:
            True if BlueStacks started successfully
        """
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

    def _connect_adb(self) -> bool:
        """Connect to ADB device.

        Returns:
            True if connection successful
        """
        logger.debug(f"Connecting to ADB at {self.device_id}...")

        try:
            # Run adb connect
            cmd = [self.adb_path, "connect", self.device_id]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10,
            )

            output = result.stdout.strip()
            logger.debug(f"ADB connect output: {output}")

            # Check for success
            if "connected" in output.lower() or "already connected" in output.lower():
                return True

            return False

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"ADB connect failed: {e}")
            return False

    def _wait_for_device(self) -> bool:
        """Wait for device to be ready.

        Returns:
            True if device became ready within timeout
        """
        start_time = time.time()

        while time.time() - start_time < self.connect_timeout:
            if self.is_connected():
                logger.info("Device connected and ready")
                return True

            # Try to connect
            self._connect_adb()
            time.sleep(1)

        logger.error(f"Device not ready after {self.connect_timeout}s timeout")
        return False

    def initialize(self) -> bool:
        """Initialize the bridge (start emulator, connect ADB).

        Call this before capturing screenshots. Automatically called
        on first capture() if not already initialized.

        Returns:
            True if initialization successful
        """
        if self._initialized:
            return True

        logger.info("Initializing ADB Screenshot Bridge...")

        # Step 1: Ensure BlueStacks is running
        if self.auto_start:
            if not self._is_bluestacks_running():
                if not self._start_bluestacks():
                    return False

                # Wait for BlueStacks to start
                logger.info("Waiting for BlueStacks to start...")
                time.sleep(5)

        # Step 2: Connect to ADB
        if not self._wait_for_device():
            logger.error(
                "Failed to connect to device. "
                "Make sure ADB is enabled in BlueStacks settings: "
                "Settings > Advanced > Android Debug Bridge (ADB)"
            )
            return False

        # Step 3: Get screen size
        self._screen_size = self._detect_screen_size()

        self._initialized = True
        logger.info(f"Initialized successfully. Screen size: {self._screen_size}")
        return True

    def _detect_screen_size(self) -> tuple[int, int]:
        """Detect screen size from device.

        Returns:
            Tuple of (width, height) in pixels
        """
        try:
            cmd = self._build_adb_command("shell", "wm", "size")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5,
            )

            # Parse output like "Physical size: 1080x1920"
            output = result.stdout.strip()
            for line in output.split("\n"):
                if "size:" in line.lower():
                    size_str = line.split(":")[-1].strip()
                    if "x" in size_str:
                        w, h = size_str.split("x")
                        return (int(w), int(h))

        except Exception as e:
            logger.warning(f"Could not detect screen size: {e}")

        # Default fallback
        return (1080, 1920)

    def capture(self) -> np.ndarray:
        """Capture screenshot via ADB screencap.

        Auto-initializes on first call if needed.

        Returns:
            RGB image as numpy array (H, W, 3)

        Raises:
            ConnectionError: If not connected to device
            RuntimeError: If screenshot capture fails
        """
        # Auto-initialize if needed
        if not self._initialized:
            if not self.initialize():
                raise ConnectionError(
                    "Failed to initialize ADB connection. "
                    "Make sure BlueStacks is running with ADB enabled."
                )

        # Capture screenshot using exec-out (binary mode)
        cmd = self._build_adb_command("exec-out", "screencap", "-p")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=10,
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"screencap failed with code {result.returncode}: "
                    f"{result.stderr.decode('utf-8', errors='ignore')}"
                )

            # Parse PNG data
            if len(result.stdout) < 100:
                raise RuntimeError("Screenshot data too small, capture may have failed")

            image = Image.open(io.BytesIO(result.stdout))
            return np.array(image.convert("RGB"))

        except subprocess.TimeoutExpired:
            raise RuntimeError("Screenshot capture timed out")
        except Exception as e:
            raise RuntimeError(f"Failed to capture screenshot: {e}")

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

    def get_screen_size(self) -> tuple[int, int]:
        """Get screen dimensions.

        Returns:
            Tuple of (width, height) in pixels
        """
        if self._screen_size is not None:
            return self._screen_size

        if not self._initialized:
            self.initialize()

        return self._screen_size or (1080, 1920)

    def disconnect(self) -> None:
        """Disconnect from ADB device."""
        try:
            cmd = [self.adb_path, "disconnect", self.device_id]
            subprocess.run(cmd, capture_output=True, timeout=5)
            logger.debug(f"Disconnected from {self.device_id}")
        except Exception as e:
            logger.warning(f"Error disconnecting: {e}")

        self._initialized = False


class MockScreenshotBridge(ScreenshotBridge):
    """Mock screenshot bridge for testing and development.

    Returns predefined or randomly generated images instead of
    actual screenshots. Useful for:
    - Unit testing without emulator
    - Development of vision components
    - Debugging training pipeline
    """

    def __init__(
        self,
        width: int = 1080,
        height: int = 1920,
        image_path: Optional[Path] = None,
    ):
        """Initialize mock bridge.

        Args:
            width: Image width to generate
            height: Image height to generate
            image_path: Optional path to a test image to return
        """
        self.width = width
        self.height = height
        self.image_path = image_path
        self._connected = True

    def capture(self) -> np.ndarray:
        """Return mock screenshot.

        Returns:
            RGB image as numpy array (H, W, 3)
        """
        if self.image_path and self.image_path.exists():
            image = Image.open(self.image_path)
            return np.array(image.convert("RGB"))

        # Generate random noise image
        return np.random.randint(
            0, 255,
            size=(self.height, self.width, 3),
            dtype=np.uint8,
        )

    def is_connected(self) -> bool:
        """Mock connection is always available."""
        return self._connected

    def set_connected(self, connected: bool) -> None:
        """Set mock connection state for testing."""
        self._connected = connected
