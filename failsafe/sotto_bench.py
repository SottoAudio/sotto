"""Entry point for the packaged Sotto Bench app (PyInstaller) and for `python sotto_bench.py`."""
import sys

from sotto_failsafe.launcher import main

if __name__ == "__main__":
    sys.exit(main())
