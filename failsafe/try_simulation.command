#!/bin/bash
# No hardware needed: runs the bench tool against a simulated Riedel frame.
cd "$(dirname "$0")"
source .venv/bin/activate
(sleep 3 && open "http://127.0.0.1:8766") &
python -m sotto_failsafe --simulate --record
