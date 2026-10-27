#!/bin/bash
# Run once. Needs Python 3.10-3.12 from python.org.
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
$PY -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt \
  && echo && echo "Installed. Try it with try_simulation.command, then edit start_bench.command for the real test."
