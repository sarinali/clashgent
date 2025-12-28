"""Emulator interaction bridge for screenshot capture and action execution."""

from .base import EmulatorBridge
from .adb import ADBBridge

__all__ = [
    "EmulatorBridge",
    "ADBBridge",
]
