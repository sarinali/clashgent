#!/bin/bash

# Android Emulator Startup Script
# This script starts your Android emulator (Android Studio or BlueStacks)

# Set Android SDK path
ANDROID_SDK="$HOME/Library/Android/sdk"
EMULATOR_PATH="$ANDROID_SDK/emulator/emulator"
BLUESTACKS_PATH="/Applications/BlueStacks.app"

# Check which emulator to use (first argument or default to android)
EMULATOR_TYPE="${1:-android}"

# Function to start Android Studio emulator
start_android_emulator() {
    if [ ! -f "$EMULATOR_PATH" ]; then
        echo "❌ Error: Android emulator not found at $EMULATOR_PATH"
        echo "Please make sure Android SDK is installed."
        exit 1
    fi

    # List available AVDs
    echo "📱 Available Android Virtual Devices:"
    "$EMULATOR_PATH" -list-avds

    # Check if any emulator is already running
    if pgrep -f "emulator.*-avd" > /dev/null || pgrep -f "qemu-system" > /dev/null; then
        echo "⚠️  An Android emulator is already running!"
        echo "If you want to start a new one, please close the existing emulator first."
        exit 0
    fi

    # Clean up any stale emulator processes
    STALE_PIDS=$(pgrep -f "crashpad_handler.*emu-crash" 2>/dev/null)
    if [ ! -z "$STALE_PIDS" ]; then
        echo "🧹 Cleaning up stale emulator processes..."
        kill $STALE_PIDS 2>/dev/null
        sleep 1
    fi

    # Default AVD name (second argument or default)
    AVD_NAME="${2:-Pixel_9}"

    echo ""
    echo "🚀 Starting Android emulator: $AVD_NAME"
    echo "This may take a moment..."

    # Start the emulator in the background
    "$EMULATOR_PATH" -avd "$AVD_NAME" &

    echo "✅ Android emulator is starting in the background..."
    echo "You can close this terminal window - the emulator will continue running."
}

# Function to start BlueStacks
start_bluestacks() {
    if [ ! -d "$BLUESTACKS_PATH" ]; then
        echo "❌ Error: BlueStacks not found at $BLUESTACKS_PATH"
        echo "Please install BlueStacks from https://www.bluestacks.com"
        exit 1
    fi

    # Check if BlueStacks is already running
    if pgrep -f "BlueStacks.app" > /dev/null; then
        echo "⚠️  BlueStacks is already running!"
        echo "Opening BlueStacks window..."
        open -a BlueStacks
        exit 0
    fi

    echo ""
    echo "🚀 Starting BlueStacks..."
    echo "This may take a moment..."

    # Start BlueStacks
    open -a BlueStacks

    echo "✅ BlueStacks is starting..."
    echo ""
    echo "💡 Note: To enable ADB debugging in BlueStacks:"
    echo "   1. Open BlueStacks Settings"
    echo "   2. Go to Advanced → Android Debug Bridge"
    echo "   3. Enable ADB"
    echo ""
    echo "Then check connection with: ~/Library/Android/sdk/platform-tools/adb devices"
}

# Main logic
case "$EMULATOR_TYPE" in
    android|as|studio)
        start_android_emulator "$@"
        ;;
    bluestacks|bs|blue)
        start_bluestacks
        ;;
    *)
        echo "Usage: $0 [android|bluestacks] [AVD_NAME]"
        echo ""
        echo "Options:"
        echo "  android, as, studio  - Start Android Studio emulator (default)"
        echo "  bluestacks, bs, blue - Start BlueStacks"
        echo ""
        echo "Examples:"
        echo "  $0                    # Start Android emulator with default AVD"
        echo "  $0 android Pixel_9    # Start Android emulator with specific AVD"
        echo "  $0 bluestacks         # Start BlueStacks"
        exit 1
        ;;
esac

