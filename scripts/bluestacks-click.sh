#!/bin/bash

# BlueStacks Click/Tap Script
# This script performs clicks/taps on BlueStacks via ADB

echo "👆 BlueStacks Click Tool"
echo "========================"
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

# Function to show usage
show_usage() {
    echo "Usage: $0 <x> <y> [options]"
    echo ""
    echo "Arguments:"
    echo "  x           X coordinate (required)"
    echo "  y           Y coordinate (required)"
    echo ""
    echo "Options:"
    echo "  --screenshot    Take a screenshot after clicking"
    echo "  --swipe x2 y2   Swipe from (x,y) to (x2,y2)"
    echo "  --hold ms       Long press for specified milliseconds"
    echo "  --test          Run a series of test clicks"
    echo ""
    echo "Examples:"
    echo "  $0 500 800                    # Click at (500, 800)"
    echo "  $0 500 800 --screenshot       # Click and take screenshot"
    echo "  $0 100 500 --swipe 400 500    # Swipe from (100,500) to (400,500)"
    echo "  $0 500 800 --hold 1000        # Long press for 1 second"
    echo "  $0 --test                     # Run test clicks"
}

# Check for test mode first
if [ "$1" == "--test" ]; then
    TEST_MODE=true
    shift
elif [ -z "$1" ] || [ -z "$2" ]; then
    if [ "$1" != "--test" ]; then
        show_usage
        exit 1
    fi
fi

# Parse coordinates if not in test mode
if [ "$TEST_MODE" != true ]; then
    X_COORD=$1
    Y_COORD=$2
    shift 2
fi

# Parse optional arguments
TAKE_SCREENSHOT=false
SWIPE_MODE=false
LONG_PRESS=false
HOLD_DURATION=0

while [ $# -gt 0 ]; do
    case "$1" in
        --screenshot)
            TAKE_SCREENSHOT=true
            shift
            ;;
        --swipe)
            SWIPE_MODE=true
            X2_COORD=$2
            Y2_COORD=$3
            shift 3
            ;;
        --hold)
            LONG_PRESS=true
            HOLD_DURATION=$2
            shift 2
            ;;
        --test)
            TEST_MODE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

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
    fi
    exit 1
fi

echo "✅ Device connected!"
echo ""

# Get screen resolution for reference
RESOLUTION=$($ADB_CMD shell wm size 2>/dev/null | grep -oE '[0-9]+x[0-9]+')
echo "📱 Screen resolution: $RESOLUTION"
echo ""

# Function to perform a tap
perform_tap() {
    local x=$1
    local y=$2
    echo "👆 Tapping at coordinates ($x, $y)..."
    if $ADB_CMD shell input tap $x $y 2>/dev/null; then
        echo "✅ Tap executed successfully"
        return 0
    else
        echo "❌ Tap failed"
        return 1
    fi
}

# Function to perform a long press
perform_long_press() {
    local x=$1
    local y=$2
    local duration=$3
    echo "👆 Long pressing at ($x, $y) for ${duration}ms..."
    if $ADB_CMD shell input swipe $x $y $x $y $duration 2>/dev/null; then
        echo "✅ Long press executed successfully"
        return 0
    else
        echo "❌ Long press failed"
        return 1
    fi
}

# Function to perform a swipe
perform_swipe() {
    local x1=$1
    local y1=$2
    local x2=$3
    local y2=$4
    echo "👆 Swiping from ($x1, $y1) to ($x2, $y2)..."
    if $ADB_CMD shell input swipe $x1 $y1 $x2 $y2 300 2>/dev/null; then
        echo "✅ Swipe executed successfully"
        return 0
    else
        echo "❌ Swipe failed"
        return 1
    fi
}

# Function to take screenshot
take_screenshot() {
    local filename="click_result_$(date +%Y%m%d_%H%M%S).png"
    echo ""
    echo "📸 Taking screenshot..."
    if $ADB_CMD exec-out screencap -p > "$filename" 2>/dev/null; then
        if file "$filename" | grep -q "PNG image"; then
            echo "✅ Screenshot saved: $filename"
        else
            echo "⚠️  Screenshot may be corrupted"
        fi
    else
        echo "❌ Failed to take screenshot"
    fi
}

# Test mode - run a series of test clicks
if [ "$TEST_MODE" == true ]; then
    echo "🧪 Running Click Tests"
    echo "======================"
    echo ""

    # Extract resolution values
    WIDTH=$(echo $RESOLUTION | cut -d'x' -f1)
    HEIGHT=$(echo $RESOLUTION | cut -d'x' -f2)

    echo "Screen dimensions: ${WIDTH}x${HEIGHT}"
    echo ""

    # Calculate center and test points
    CENTER_X=$((WIDTH / 2))
    CENTER_Y=$((HEIGHT / 2))

    echo "Test 1: Tap center of screen ($CENTER_X, $CENTER_Y)"
    perform_tap $CENTER_X $CENTER_Y
    sleep 0.5
    echo ""

    echo "Test 2: Tap top-left area (100, 100)"
    perform_tap 100 100
    sleep 0.5
    echo ""

    echo "Test 3: Tap bottom-center ($CENTER_X, $((HEIGHT - 200)))"
    perform_tap $CENTER_X $((HEIGHT - 200))
    sleep 0.5
    echo ""

    echo "Test 4: Long press center for 500ms"
    perform_long_press $CENTER_X $CENTER_Y 500
    sleep 0.5
    echo ""

    echo "Test 5: Swipe from left to right (scroll gesture)"
    perform_swipe 100 $CENTER_Y $((WIDTH - 100)) $CENTER_Y
    sleep 0.5
    echo ""

    echo "Test 6: Swipe up (scroll down gesture)"
    perform_swipe $CENTER_X $((HEIGHT - 300)) $CENTER_X 300
    sleep 0.5
    echo ""

    echo "🎉 All tests completed!"
    echo ""

    read -p "Take a final screenshot to verify? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        take_screenshot
    fi

    exit 0
fi

# Execute the requested action
if [ "$SWIPE_MODE" == true ]; then
    perform_swipe $X_COORD $Y_COORD $X2_COORD $Y2_COORD
elif [ "$LONG_PRESS" == true ]; then
    perform_long_press $X_COORD $Y_COORD $HOLD_DURATION
else
    perform_tap $X_COORD $Y_COORD
fi

# Take screenshot if requested
if [ "$TAKE_SCREENSHOT" == true ]; then
    sleep 0.3  # Brief delay to let UI update
    take_screenshot
fi

echo ""
echo "Done!"
