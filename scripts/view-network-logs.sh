#!/bin/bash

# Android Emulator Network Logs Viewer
# This script helps you view network requests and API calls from your Android emulator

# Set Android SDK path
ANDROID_SDK="$HOME/Library/Android/sdk"
ADB_PATH="$ANDROID_SDK/platform-tools/adb"

# Check if adb exists (try PATH first, then default location)
if command -v adb &> /dev/null; then
    ADB_CMD="adb"
else
    ADB_CMD="$ADB_PATH"
    if [ ! -f "$ADB_CMD" ]; then
        echo "❌ Error: ADB not found"
        echo "Please install Android SDK platform-tools or add it to your PATH"
        exit 1
    fi
fi

# Check if device is connected (Android Studio emulator or BlueStacks)
CONNECTED_DEVICES=$("$ADB_CMD" devices | grep -v "List of devices" | grep -E "device$|emulator" | wc -l | tr -d ' ')

if [ "$CONNECTED_DEVICES" -eq 0 ]; then
    echo "⚠️  No Android device/emulator detected via ADB"
    echo ""
    echo "For Android Studio emulator:"
    echo "  Run: ./start-emulator.sh"
    echo ""
    echo "For BlueStacks:"
    echo "  1. Enable ADB in BlueStacks Settings → Advanced"
    echo "  2. Connect: $ADB_CMD connect 127.0.0.1:5555"
    echo ""
    echo "Current ADB devices:"
    "$ADB_CMD" devices
    exit 1
fi

# Try to connect to BlueStacks if not already connected
if ! "$ADB_CMD" devices | grep -q "127.0.0.1:5555"; then
    if pgrep -f "BlueStacks.app" > /dev/null; then
        echo "🔗 Connecting to BlueStacks..."
        "$ADB_CMD" connect 127.0.0.1:5555 2>/dev/null
        sleep 1
    fi
fi

echo "📡 Android Device/Emulator Network Logs Viewer"
echo "=============================================="
echo ""
echo "Select an option:"
echo "1) View all network-related logs (HTTP, HTTPS, DNS, etc.)"
echo "2) View HTTP/HTTPS requests only"
echo "3) Monitor network traffic with tcpdump (requires root)"
echo "4) View network stats"
echo "5) View DNS queries"
echo "6) View all logcat logs"
echo "7) Follow logs in real-time (Ctrl+C to stop)"
echo ""
read -p "Enter option (1-7): " option

case $option in
    1)
        echo ""
        echo "📊 Viewing all network-related logs..."
        echo "Press Ctrl+C to stop"
        echo ""
        "$ADB_CMD" logcat | grep -iE "(http|https|network|dns|tcp|udp|socket|connection|api|request|response)"
        ;;
    2)
        echo ""
        echo "🌐 Viewing HTTP/HTTPS requests..."
        echo "Press Ctrl+C to stop"
        echo ""
        "$ADB_CMD" logcat | grep -iE "(http|https|okhttp|retrofit|volley|urlconnection)"
        ;;
    3)
        echo ""
        echo "🔍 Capturing network traffic with tcpdump..."
        echo "This requires root access on the emulator"
        echo "Press Ctrl+C to stop"
        echo ""
        "$ADB_CMD" shell "su -c 'tcpdump -i any -s 0 -w /sdcard/capture.pcap'" &
        TCPDUMP_PID=$!
        echo "Capturing... Press Ctrl+C to stop and save"
        wait $TCPDUMP_PID
        "$ADB_CMD" pull /sdcard/capture.pcap ./network-capture.pcap
        echo "✅ Network capture saved to network-capture.pcap"
        echo "Open with Wireshark or tcpdump: tcpdump -r network-capture.pcap"
        ;;
    4)
        echo ""
        echo "📈 Network statistics:"
        echo ""
        "$ADB_CMD" shell "cat /proc/net/dev"
        echo ""
        echo "Active connections:"
        "$ADB_CMD" shell "netstat -an" 2>/dev/null || "$ADB_CMD" shell "ss -an" 2>/dev/null || echo "netstat/ss not available"
        ;;
    5)
        echo ""
        echo "🔍 Viewing DNS queries..."
        echo "Press Ctrl+C to stop"
        echo ""
        "$ADB_CMD" logcat | grep -iE "(dns|getaddrinfo|gethostbyname)"
        ;;
    6)
        echo ""
        echo "📋 Viewing all logcat logs..."
        echo "Press Ctrl+C to stop"
        echo ""
        "$ADB_CMD" logcat
        ;;
    7)
        echo ""
        echo "👀 Following logs in real-time..."
        echo "Press Ctrl+C to stop"
        echo ""
        "$ADB_CMD" logcat -v time | grep -iE "(http|https|network|api|request|response)"
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

