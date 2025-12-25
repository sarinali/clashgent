#!/bin/bash

# Android Emulator Crash Logs Viewer
# This script helps you view crash logs and diagnose app crashes

# Set Android SDK path
ANDROID_SDK="$HOME/Library/Android/sdk"
ADB_PATH="$ANDROID_SDK/platform-tools/adb"

# Check if adb exists
if [ ! -f "$ADB_PATH" ]; then
    echo "❌ Error: ADB not found at $ADB_PATH"
    echo "Please make sure Android SDK platform-tools are installed."
    exit 1
fi

# Check if emulator is running
if ! "$ADB_PATH" devices | grep -q "emulator"; then
    echo "⚠️  No emulator detected. Please start an emulator first."
    echo "Run: ./start-emulator.sh"
    exit 1
fi

echo "💥 Android Emulator Crash Logs Viewer"
echo "======================================"
echo ""
echo "Select an option:"
echo "1) View recent crashes (FATAL EXCEPTION)"
echo "2) View crashes for a specific app"
echo "3) View all crash logs (AndroidRuntime errors)"
echo "4) View dropbox crash reports"
echo "5) Follow crash logs in real-time"
echo "6) View native crashes"
echo "7) Check emulator architecture (for compatibility)"
echo ""
read -p "Enter option (1-7): " option

case $option in
    1)
        echo ""
        echo "💥 Recent crashes (FATAL EXCEPTION):"
        echo "====================================="
        echo ""
        "$ADB_PATH" logcat -d | grep -A 30 "FATAL EXCEPTION" | tail -100
        ;;
    2)
        echo ""
        read -p "Enter app package name (e.g., com.supercell.clashroyale): " package_name
        echo ""
        echo "💥 Crashes for $package_name:"
        echo "=============================="
        echo ""
        "$ADB_PATH" logcat -d | grep -A 30 "FATAL EXCEPTION.*$package_name" | tail -100
        ;;
    3)
        echo ""
        echo "💥 All AndroidRuntime errors:"
        echo "=============================="
        echo ""
        "$ADB_PATH" logcat -d -s AndroidRuntime:E | tail -200
        ;;
    4)
        echo ""
        echo "📦 Dropbox crash reports:"
        echo "========================="
        echo ""
        "$ADB_PATH" shell dumpsys dropbox | grep -i crash | head -30
        echo ""
        echo "To view a specific crash report, run:"
        echo "  $ADB_PATH shell dumpsys dropbox --print <timestamp>"
        ;;
    5)
        echo ""
        echo "👀 Following crash logs in real-time..."
        echo "Press Ctrl+C to stop"
        echo ""
        "$ADB_PATH" logcat | grep -iE "(FATAL|AndroidRuntime|crash|exception)"
        ;;
    6)
        echo ""
        echo "🔧 Native crashes:"
        echo "=================="
        echo ""
        "$ADB_PATH" logcat -d | grep -iE "(SIGSEGV|SIGABRT|SIGFPE|tombstone|native crash)" | tail -100
        ;;
    7)
        echo ""
        echo "🏗️  Emulator Architecture Information:"
        echo "======================================"
        echo ""
        echo "CPU ABI: $("$ADB_PATH" shell getprop ro.product.cpu.abi)"
        echo "CPU ABI List: $("$ADB_PATH" shell getprop ro.product.cpu.abilist)"
        echo "Device Model: $("$ADB_PATH" shell getprop ro.product.model)"
        echo "Android Version: $("$ADB_PATH" shell getprop ro.build.version.release)"
        echo "SDK Version: $("$ADB_PATH" shell getprop ro.build.version.sdk)"
        echo ""
        echo "💡 Tip: Some apps require specific CPU architectures."
        echo "   If your app crashes with 'nativeLoad' errors, it might be an architecture mismatch."
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

