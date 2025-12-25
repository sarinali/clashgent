# Approach Decision: Screenshots vs Network Traffic

## Decision: Use Screenshots via ADB

Using screenshot-based computer vision instead of network traffic decoding.

## Why Not Network Traffic?

Tried multiple approaches:

- Network log viewing scripts (`view-network-logs.sh`, `view-app-network-logs.sh`)
- Wireshark/tshark packet analysis
- Charles Proxy interception
- Captured raw traffic to `.pcap` files in `captures/`

**Problem**: All captured traffic is encrypted (Sodium/NaCl). Raw dumps show binary data but no readable game state.

- **Complex protocol**: Custom binary over TCP (port 9339), encrypted with Sodium/NaCl
- **Reverse engineering required**: See [Frida script](https://gist.github.com/iGio90/7656f3719ad8fe278dce6bfc09bdf439) and [protocol docs](https://github.com/clugh/cocdp/wiki/Protocol)
- **Gets patched**: Supercell updates protocol regularly, breaking reverse-engineered solutions
- **Detection risk**: Network interception may trigger anti-cheat

## Why Screenshots?

- Stable (doesn't break with protocol updates)
- Accessible via `adb shell screencap` (no root needed)
- No detection risk
- Contains all visible game state needed

Screenshots via ADB bridge should be sufficient for training.
