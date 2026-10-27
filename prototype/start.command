#!/bin/bash
# Edit the channel numbers to match your patch, then double-click to start.
# Find your interface name and channel counts with:  .venv/bin/python rt.py --list
cd "$(dirname "$0")"
source .venv/bin/activate
(sleep 6 && open "http://127.0.0.1:8765") &
python rt.py --device "Scarlett" \
  --mic 1 --names caller \
  --ref 3,4 --ref-names pa_l,pa_r \
  --out 1 --out-dry 2 --monitor 3,4 \
  --record
