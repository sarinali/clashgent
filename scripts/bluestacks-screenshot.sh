#!/bin/bash

# BlueStacks Screenshot Script
# This script takes a screenshot from BlueStacks via ADB

echo "📸 BlueStacks Screenshot Tool"
echo "=============================="
echo ""

# Set ADB path (use PATH if available, otherwise use default location)
if command -v adb &> /dev/null; then
    ADB_CMD="adb"
else
    ADB_CMD="$HOME/Library/Android/sdk/platform-tools/adb"
    if [ ! -f "$ADB_CMD" ]; then
        echo "❌ ADB not found"
        echo "Please install Android SDK platform-tools or add it to your PATH"
        exit 1
    fi
fi

# Check if BlueStacks is running
if ! pgrep -f "BlueStacks.app" > /dev/null; then
    echo "⚠️  BlueStacks is not running!"
    echo "Starting BlueStacks..."
    open -a BlueStacks
    echo "Please wait for BlueStacks to start, then run this script again."
    exit 1
fi

# Check ADB connection
echo "Checking ADB connection..."

# Try to connect to BlueStacks default port (127.0.0.1:5555)
echo "Connecting to BlueStacks ADB (127.0.0.1:5555)..."
$ADB_CMD connect 127.0.0.1:5555 2>/dev/null

# Wait a moment for connection
sleep 1

# Check if any devices are connected
DEVICES=$($ADB_CMD devices | grep -v "List of devices" | grep -E "device$|unauthorized" | wc -l | tr -d ' ')
CONNECTED_DEVICES=$($ADB_CMD devices | grep -v "List of devices" | grep "device$" | wc -l | tr -d ' ')

if [ "$CONNECTED_DEVICES" -eq 0 ]; then
    echo ""
    echo "❌ No devices connected via ADB"
    echo ""
    echo "Current ADB devices:"
    $ADB_CMD devices
    echo ""
    
    if [ "$DEVICES" -gt 0 ]; then
        echo "⚠️  Device found but may be unauthorized"
        echo "Check BlueStacks for an authorization prompt"
    else
        echo "📋 To enable ADB in BlueStacks:"
        echo "   1. Open BlueStacks"
        echo "   2. Click the gear icon (Settings) in the top right"
        echo "   3. Go to 'Advanced' tab"
        echo "   4. Scroll down and enable 'Android Debug Bridge (ADB)'"
        echo "   5. Make sure it shows: Connect to Android at 127.0.0.1:5555"
        echo ""
        echo "If ADB is enabled but not connecting, try:"
        echo "   adb connect 127.0.0.1:5555"
    fi
    exit 1
fi

echo "✅ Device connected!"
echo ""

# Get output filename (optional argument or default)
OUTPUT_FILE="${1:-bluestacks_screenshot_$(date +%Y%m%d_%H%M%S).png}"

echo "📸 Taking screenshot..."

# Use exec-out for binary data (handles PNG correctly without line ending issues)
if $ADB_CMD exec-out screencap -p > "$OUTPUT_FILE" 2>/dev/null; then
    # Verify it's a valid PNG file
    if ! file "$OUTPUT_FILE" | grep -q "PNG image"; then
        echo "⚠️  Warning: Screenshot may be corrupted, trying alternative method..."
        # Fallback: use shell method and fix line endings properly
        $ADB_CMD shell screencap -p | sed 's/\r$//' > "$OUTPUT_FILE" 2>/dev/null
    fi
    
    FILE_SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    
    # Verify PNG is valid
    if file "$OUTPUT_FILE" | grep -q "PNG image"; then
        echo "✅ Screenshot saved: $OUTPUT_FILE ($FILE_SIZE)"
        echo ""
        echo "💡 To view: open $OUTPUT_FILE"
        
        # Try to open the screenshot
        if command -v open &> /dev/null; then
            read -p "Open screenshot now? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                open "$OUTPUT_FILE"
            fi
        fi
    else
        echo "❌ Screenshot file appears corrupted"
        echo "File type: $(file "$OUTPUT_FILE")"
        exit 1
    fi
else
    echo "❌ Failed to take screenshot"
    echo "Make sure BlueStacks is running and ADB is enabled"
    exit 1
fi

