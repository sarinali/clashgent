#!/bin/bash

# Clash Royale Traffic Capture & Analysis Script
# Captures game protocol traffic on port 9339

CAPTURE_DIR="/Users/sarinali/Projects/clashgent/captures"
mkdir -p "$CAPTURE_DIR"

echo "🎮 Clash Royale Traffic Capture"
echo "================================"
echo ""

# Detect network interface
DEFAULT_IF=$(route -n get default 2>/dev/null | grep 'interface:' | awk '{print $2}')
[ -z "$DEFAULT_IF" ] && DEFAULT_IF="en0"

echo "📡 Detected interface: $DEFAULT_IF"
echo ""

# Check for Wireshark/tshark
HAS_TSHARK=false
HAS_WIRESHARK=false
command -v tshark &>/dev/null && HAS_TSHARK=true
[ -d "/Applications/Wireshark.app" ] && HAS_WIRESHARK=true

echo "Tools available:"
echo "  tcpdump: ✅"
$HAS_TSHARK && echo "  tshark: ✅" || echo "  tshark: ❌ (install: brew install wireshark)"
$HAS_WIRESHARK && echo "  Wireshark: ✅" || echo "  Wireshark: ❌"
echo ""

# Check current Clash Royale connections
echo "🔍 Checking for active Clash Royale connections..."
CR_UID=$(adb shell "pm list packages -U 2>/dev/null | grep -i 'clashroyale'" | grep -oE "uid:[0-9]+" | cut -d: -f2 | head -1)

if [ ! -z "$CR_UID" ]; then
    echo "  Clash Royale UID: $CR_UID"

    # Decode and show connections
    adb shell "cat /proc/net/tcp /proc/net/tcp6" 2>/dev/null | awk -v uid="$CR_UID" '$8 == uid {print $3}' | while read hex_addr; do
        addr=$(echo $hex_addr | cut -d: -f1)
        port=$(echo $hex_addr | cut -d: -f2)
        ip=$(printf "%d.%d.%d.%d" 0x${addr:6:2} 0x${addr:4:2} 0x${addr:2:2} 0x${addr:0:2})
        port_dec=$((16#$port))
        echo "  Connected to: $ip:$port_dec"
    done
else
    echo "  ⚠️  Clash Royale not detected or not running"
fi
echo ""

# Menu
echo "Select option:"
echo "1) Capture to .pcap file (recommended)"
echo "2) Live capture with hex dump"
echo "3) Live capture with tshark (detailed)"
echo "4) Analyze existing .pcap file"
echo "5) Monitor connections in real-time"
echo ""
read -p "Enter option (1-5): " option

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PCAP_FILE="$CAPTURE_DIR/clash_${TIMESTAMP}.pcap"

case $option in
    1)
        echo ""
        echo "📁 Saving to: $PCAP_FILE"
        echo "🔴 Capturing on port 9339... (Ctrl+C to stop)"
        echo ""
        echo "Play the game to generate traffic!"
        echo ""
        sudo tcpdump -i "$DEFAULT_IF" -w "$PCAP_FILE" 'port 9339' -v

        echo ""
        echo "✅ Capture saved to: $PCAP_FILE"
        echo ""

        # Quick analysis
        if [ -f "$PCAP_FILE" ]; then
            PACKET_COUNT=$(tcpdump -r "$PCAP_FILE" 2>/dev/null | wc -l | tr -d ' ')
            echo "📊 Captured $PACKET_COUNT packets"

            if $HAS_WIRESHARK; then
                read -p "Open in Wireshark? (y/n): " open_ws
                [ "$open_ws" = "y" ] && open -a Wireshark "$PCAP_FILE"
            fi
        fi
        ;;

    2)
        echo ""
        echo "🔴 Live capture with hex dump (Ctrl+C to stop)..."
        echo ""
        sudo tcpdump -i "$DEFAULT_IF" -X -nn 'port 9339'
        ;;

    3)
        if ! $HAS_TSHARK; then
            echo "❌ tshark not installed. Run: brew install wireshark"
            exit 1
        fi
        echo ""
        echo "🔴 Live capture with tshark (Ctrl+C to stop)..."
        echo ""
        sudo tshark -i "$DEFAULT_IF" -f 'port 9339' -V
        ;;

    4)
        echo ""
        echo "Available captures:"
        ls -la "$CAPTURE_DIR"/*.pcap 2>/dev/null || echo "  No captures found"
        echo ""
        read -p "Enter pcap file path: " pcap_path

        if [ ! -f "$pcap_path" ]; then
            echo "❌ File not found"
            exit 1
        fi

        echo ""
        echo "📊 Analyzing $pcap_path..."
        echo ""

        # Basic stats
        echo "=== Packet Summary ==="
        tcpdump -r "$pcap_path" -nn 2>/dev/null | head -20

        echo ""
        echo "=== Conversation Stats ==="
        tcpdump -r "$pcap_path" -nn 2>/dev/null | awk '{print $3, $5}' | sort | uniq -c | sort -rn | head -10

        echo ""
        echo "=== First Packet Payload (hex) ==="
        tcpdump -r "$pcap_path" -X 2>/dev/null | head -40

        if $HAS_WIRESHARK; then
            echo ""
            read -p "Open in Wireshark? (y/n): " open_ws
            [ "$open_ws" = "y" ] && open -a Wireshark "$pcap_path"
        fi
        ;;

    5)
        echo ""
        echo "🔄 Monitoring Clash Royale connections (Ctrl+C to stop)..."
        echo ""

        while true; do
            clear
            echo "🎮 Clash Royale Connection Monitor"
            echo "=================================="
            echo "$(date)"
            echo ""

            if [ ! -z "$CR_UID" ]; then
                echo "TCP Connections (UID: $CR_UID):"
                adb shell "cat /proc/net/tcp /proc/net/tcp6" 2>/dev/null | awk -v uid="$CR_UID" '
                    $8 == uid && $4 == "01" {
                        split($2, local, ":")
                        split($3, remote, ":")

                        # Convert hex IP to decimal (little-endian)
                        lip = sprintf("%d.%d.%d.%d",
                            strtonum("0x" substr(local[1],7,2)),
                            strtonum("0x" substr(local[1],5,2)),
                            strtonum("0x" substr(local[1],3,2)),
                            strtonum("0x" substr(local[1],1,2)))
                        lport = strtonum("0x" local[2])

                        rip = sprintf("%d.%d.%d.%d",
                            strtonum("0x" substr(remote[1],7,2)),
                            strtonum("0x" substr(remote[1],5,2)),
                            strtonum("0x" substr(remote[1],3,2)),
                            strtonum("0x" substr(remote[1],1,2)))
                        rport = strtonum("0x" remote[2])

                        printf "  %s:%d -> %s:%d\n", lip, lport, rip, rport
                    }'
            else
                echo "⚠️  Clash Royale not detected"
            fi

            sleep 2
        done
        ;;

    *)
        echo "❌ Invalid option"
        exit 1
        ;;
esac
