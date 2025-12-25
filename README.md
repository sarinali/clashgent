# Clashgent: Training a model to play Clash Royale autonomously

I wanted to play 2v2 Clash with my boyfriend but I realized I don't have enough cards and it is too time consuming to play Clash Royale on my own solo, so I'm going to train a neural network to play for me, and run this in the background.

## Overview

Currently using:

- Bluestacks (emulator, gets around Clash Royale emulator detection)

## Project Structure

```
clashgent/
├── captures/          # Network capture files (.pcap)
├── scripts/           # Utility scripts
│   ├── bluestacks-screenshot.sh
│   ├── capture-clash-traffic.sh
│   ├── start-emulator.sh
│   ├── view-app-network-logs.sh
│   ├── view-crash-logs.sh
│   └── view-network-logs.sh
└── README.md
```
