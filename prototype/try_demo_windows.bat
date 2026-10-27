@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
start "" http://127.0.0.1:8765
python rt.py --play demo\demo_trackmap.json --mic 1 --ref 1
pause
