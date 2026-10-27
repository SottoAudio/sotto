#!/bin/bash
# Fail-safe bench test. Edit the device and channels to match your patch, then double-click.
# List devices: .venv/bin/python -m sotto_failsafe --list
cd "$(dirname "$0")"
source .venv/bin/activate
(sleep 3 && open "http://127.0.0.1:8766") &
python -m sotto_failsafe --device "Dante Virtual Soundcard" \
  --talker-in 1-4 --talker-out 1-4 \
  --probe-in 5 --probe-out 5 --heartbeat-out 6 \
  --listen-in 6 --monitor 7,8 --record
