"""Entry point for the packaged Sotto Voce Bench app (PyInstaller) and for `python sotto_bench.py`."""
import multiprocessing
import sys

from sotto_failsafe.launcher import main

if __name__ == "__main__":
    multiprocessing.freeze_support()     # the PA delay finder runs in a helper process
    sys.exit(main())
