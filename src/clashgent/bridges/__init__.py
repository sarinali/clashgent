"""Emulator interaction bridges for screenshot capture and action execution."""

from .base import ScreenshotBridge, ActionBridge
from .screenshot import ADBScreenshotBridge
from .action import ADBActionBridge

__all__ = [
    "ScreenshotBridge",
    "ActionBridge",
    "ADBScreenshotBridge",
    "ADBActionBridge",
]
