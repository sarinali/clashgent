#!/bin/bash

# App-Specific Network Logs Viewer
# Shows network requests with app/package name identification

# Set Android SDK path
ANDROID_SDK="$HOME/Library/Android/sdk"
ADB_PATH="$ANDROID_SDK/platform-tools/adb"

# Check if adb exists
if command -v adb &> /dev/null; then
    ADB_CMD="adb"
else
    ADB_CMD="$ADB_PATH"
    if [ ! -f "$ADB_CMD" ]; then
        echo "❌ Error: ADB not found"
        exit 1
    fi
fi

# Check if device is connected
CONNECTED_DEVICES=$("$ADB_CMD" devices | grep -v "List of devices" | grep -E "device$|emulator" | wc -l | tr -d ' ')
if [ "$CONNECTED_DEVICES" -eq 0 ]; then
    # Try connecting to BlueStacks
    if pgrep -f "BlueStacks.app" > /dev/null; then
        "$ADB_CMD" connect 127.0.0.1:5555 2>/dev/null
        sleep 1
    fi
    CONNECTED_DEVICES=$("$ADB_CMD" devices | grep -v "List of devices" | grep -E "device$|emulator" | wc -l | tr -d ' ')
    if [ "$CONNECTED_DEVICES" -eq 0 ]; then
        echo "❌ No device connected"
        exit 1
    fi
fi

echo "📱 App Network Activity Viewer"
echo "=============================="
echo ""

# Function to get package name from UID
get_package_from_uid() {
    local uid=$1
    # Remove user ID part (e.g., 10123 -> 123)
    local app_uid=$(echo $uid | sed 's/^[0-9]*//')
    if [ -z "$app_uid" ] || [ "$app_uid" = "-1" ]; then
        echo "system"
        return
    fi
    # Get package name from UID
    "$ADB_CMD" shell "dumpsys package | grep -B 2 \"userId=$uid\" | grep 'Package \[' | head -1 | sed 's/.*Package \[\(.*\)\].*/\1/'" 2>/dev/null | head -1
}

# Function to show running apps
show_running_apps() {
    echo "📋 Apps currently running:"
    echo ""

    # Method 1: Get running app processes using dumpsys activity
    echo "🔄 Active App Processes:"
    "$ADB_CMD" shell "dumpsys activity processes | grep -E 'ProcessRecord\{' | sed 's/.*ProcessRecord{[^ ]* [0-9]*:\([^/]*\).*/\1/' | sort -u" 2>/dev/null | while read -r pkg; do
        if [ ! -z "$pkg" ]; then
            echo "  • $pkg"
        fi
    done

    echo ""
    echo "🎮 Foreground/Recent Apps:"
    "$ADB_CMD" shell "dumpsys activity activities | grep -E 'mResumedActivity|topResumedActivity|mFocusedApp'" 2>/dev/null | head -5

    echo ""
    echo "📦 Third-party apps installed:"
    "$ADB_CMD" shell "pm list packages -3" 2>/dev/null | sed 's/package:/  • /' | head -15
    echo ""
}

# Function to monitor network by app
monitor_by_app() {
    local app_package=$1
    echo "🔍 Monitoring network activity for: $app_package"
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Get UID for the package
    local uid=$("$ADB_CMD" shell "dumpsys package $app_package | grep userId | head -1 | sed 's/.*userId=\([0-9]*\).*/\1/'" 2>/dev/null)
    
    if [ -z "$uid" ]; then
        echo "⚠️  Could not find UID for $app_package"
        echo "Showing all network logs filtered by package name..."
        "$ADB_CMD" logcat | grep -i "$app_package" | grep -iE "(http|https|network|api|request|response|socket|connection)"
    else
        echo "📦 Package: $app_package (UID: $uid)"
        echo ""
        "$ADB_CMD" logcat | grep -E "(uid=$uid|$app_package)" | grep -iE "(http|https|network|api|request|response|socket|connection|TrafficStats)"
    fi
}

# Function to show network stats by app
show_network_stats() {
    echo "📊 Network Statistics by App:"
    echo ""

    # Get UID to package mapping first
    echo "📦 Getting app network usage..."
    echo ""

    # Use dumpsys netstats for detailed per-UID stats
    "$ADB_CMD" shell "dumpsys netstats --uid" 2>/dev/null | grep -A 5 "uid=" | head -50

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📈 Per-App Network Usage (Top 10):"
    echo ""

    # Get netstats detail and parse it
    "$ADB_CMD" shell "dumpsys netstats detail" 2>/dev/null | grep -E "^  uid=" | head -10 | while read -r line; do
        uid=$(echo "$line" | grep -oE "uid=[0-9]+" | cut -d= -f2)
        rx=$(echo "$line" | grep -oE "rxBytes=[0-9]+" | cut -d= -f2)
        tx=$(echo "$line" | grep -oE "txBytes=[0-9]+" | cut -d= -f2)

        if [ ! -z "$uid" ] && [ "$uid" -gt 10000 ]; then
            # Get package name for this UID
            pkg=$("$ADB_CMD" shell "pm list packages -U | grep \"uid:$uid\"" 2>/dev/null | cut -d: -f2 | cut -d' ' -f1)
            if [ ! -z "$pkg" ]; then
                rx_mb=$((rx / 1024 / 1024))
                tx_mb=$((tx / 1024 / 1024))
                printf "  %-40s RX: %4d MB  TX: %4d MB\n" "$pkg" "$rx_mb" "$tx_mb"
            fi
        fi
    done

    echo ""
    echo "💡 Note: Stats show data since last device boot"
}

# Function to monitor all apps with UID mapping
monitor_all_with_apps() {
    echo "👀 Monitoring all network activity with app identification..."
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Create UID to package mapping cache
    declare -A uid_cache
    
    "$ADB_CMD" logcat -v time | while IFS= read -r line; do
        # Check for TrafficStats entries with UID
        if echo "$line" | grep -qE "TrafficStats.*statsUid="; then
            uid=$(echo "$line" | grep -oE "statsUid=[0-9-]+" | cut -d= -f2)
            if [ ! -z "$uid" ] && [ "$uid" != "-1" ]; then
                if [ -z "${uid_cache[$uid]}" ]; then
                    pkg=$(get_package_from_uid $uid)
                    uid_cache[$uid]=$pkg
                fi
                pkg="${uid_cache[$uid]}"
                echo "[$pkg] $line"
            else
                echo "[system] $line"
            fi
        # Check for package-specific logs
        elif echo "$line" | grep -qE "(http|https|network|api|request|response)"; then
            # Try to extract package name from log tag
            pkg=$(echo "$line" | grep -oE "[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*" | head -1)
            if [ ! -z "$pkg" ]; then
                echo "[$pkg] $line"
            else
                echo "$line"
            fi
        fi
    done
}

# Function to show active connections per app
show_active_connections() {
    echo "🌐 Active Network Connections by App:"
    echo ""

    # Get all TCP connections and their UIDs
    echo "TCP Connections:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    "$ADB_CMD" shell "cat /proc/net/tcp /proc/net/tcp6" 2>/dev/null | tail -n +2 | while read -r line; do
        uid=$(echo "$line" | awk '{print $8}')
        local_addr=$(echo "$line" | awk '{print $2}')
        remote_addr=$(echo "$line" | awk '{print $3}')
        state=$(echo "$line" | awk '{print $4}')

        # Only show established connections (state 01)
        if [ "$state" = "01" ] && [ ! -z "$uid" ] && [ "$uid" -gt 10000 ]; then
            pkg=$("$ADB_CMD" shell "pm list packages -U 2>/dev/null | grep \"uid:$uid\"" | head -1 | sed 's/package:\([^ ]*\).*/\1/')
            if [ ! -z "$pkg" ]; then
                # Convert hex addresses to readable format
                echo "  [$pkg] $local_addr -> $remote_addr"
            fi
        fi
    done | head -20

    echo ""
    echo "UDP Connections:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    "$ADB_CMD" shell "cat /proc/net/udp /proc/net/udp6" 2>/dev/null | tail -n +2 | while read -r line; do
        uid=$(echo "$line" | awk '{print $8}')
        if [ ! -z "$uid" ] && [ "$uid" -gt 10000 ]; then
            pkg=$("$ADB_CMD" shell "pm list packages -U 2>/dev/null | grep \"uid:$uid\"" | head -1 | sed 's/package:\([^ ]*\).*/\1/')
            if [ ! -z "$pkg" ]; then
                echo "  [$pkg] UDP socket active"
            fi
        fi
    done | head -10 | sort -u

    echo ""
}

# Function to monitor a specific app's connections in real-time
monitor_app_realtime() {
    local app_package=$1

    # Get UID for the package
    local uid=$("$ADB_CMD" shell "pm list packages -U | grep \"$app_package\"" 2>/dev/null | grep -oE "uid:[0-9]+" | cut -d: -f2)

    if [ -z "$uid" ]; then
        echo "⚠️  Could not find UID for $app_package"
        echo "Available packages matching '$app_package':"
        "$ADB_CMD" shell "pm list packages | grep -i '$app_package'" 2>/dev/null
        return 1
    fi

    echo "📦 Package: $app_package"
    echo "🔢 UID: $uid"
    echo ""
    echo "🔄 Monitoring connections (Ctrl+C to stop)..."
    echo ""

    while true; do
        clear
        echo "📱 Live Network Monitor: $app_package (UID: $uid)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # Show TCP connections for this UID
        echo "TCP Connections:"
        "$ADB_CMD" shell "cat /proc/net/tcp /proc/net/tcp6" 2>/dev/null | awk -v uid="$uid" '$8 == uid && $4 == "01" {print "  " $2 " -> " $3}'

        echo ""
        echo "UDP Sockets:"
        "$ADB_CMD" shell "cat /proc/net/udp /proc/net/udp6" 2>/dev/null | awk -v uid="$uid" '$8 == uid {print "  " $2}'

        echo ""
        echo "Last updated: $(date '+%H:%M:%S')"
        sleep 2
    done
}

# Main menu
echo "Select an option:"
echo "1) Show running apps"
echo "2) Monitor network for specific app (by package name)"
echo "3) Monitor all network activity with app identification"
echo "4) Show network statistics by app"
echo "5) View HTTP/HTTPS requests with app names"
echo "6) Show active connections per app"
echo "7) Real-time connection monitor for specific app"
echo ""
read -p "Enter option (1-7): " option

case $option in
    1)
        show_running_apps
        ;;
    2)
        echo ""
        read -p "Enter app package name (e.g., com.supercell.clashroyale): " app_package
        echo ""
        monitor_by_app "$app_package"
        ;;
    3)
        echo ""
        monitor_all_with_apps
        ;;
    4)
        echo ""
        show_network_stats
        ;;
    5)
        echo ""
        echo "🌐 HTTP/HTTPS requests with app identification:"
        echo "Press Ctrl+C to stop"
        echo ""
        "$ADB_CMD" logcat -v time | grep -iE "(http|https|okhttp|retrofit|volley|urlconnection)" | while IFS= read -r line; do
            # Try to extract package name
            pkg=$(echo "$line" | grep -oE "[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*" | head -1)
            if [ ! -z "$pkg" ]; then
                echo "[$pkg] $line"
            else
                echo "$line"
            fi
        done
        ;;
    6)
        echo ""
        show_active_connections
        ;;
    7)
        echo ""
        read -p "Enter app package name (e.g., com.supercell.clashroyale): " app_package
        echo ""
        monitor_app_realtime "$app_package"
        ;;
    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac

