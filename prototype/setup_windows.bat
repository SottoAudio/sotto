@echo off
REM Run once to install. Needs Python 3.10-3.12 from python.org ("Add to PATH" ticked).
cd /d "%~dp0"
py -3 -m venv .venv || python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
python tools\get_models.py
python rt.py --selftest
echo.
echo Installed. Edit start_windows.bat, then double-click it.
pause
