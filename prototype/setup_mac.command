#!/bin/bash
# Double-click (or run in Terminal) once to install. Needs Python 3.10-3.12 from python.org.
cd "$(dirname "$0")"
PY=$(command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
echo "Using $PY ($($PY --version))"
$PY -m venv .venv && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt \
  && python tools/get_models.py && python rt.py --selftest && echo && echo "Installed. Start with: ./start.command  (edit it first)"
