#!/bin/bash
# Plays the synthetic stadium demo through the processing to your laptop headphones. No interface needed.
cd "$(dirname "$0")"
source .venv/bin/activate
(sleep 6 && open "http://127.0.0.1:8765") &
python rt.py --play demo/demo_trackmap.json --mic 1 --ref 1
