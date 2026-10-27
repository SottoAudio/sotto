@echo off
REM Edit the device and channel numbers, then double-click. List devices with: .venv\Scripts\python rt.py --list
REM Pick the "Windows WASAPI" entry for your interface (use its index number).
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" http://127.0.0.1:8765
python rt.py --device 12 --mic 1 --names caller --ref 3,4 --ref-names pa_l,pa_r --out 1 --out-dry 2 --monitor 3,4 --record --latency 0.02
pause
