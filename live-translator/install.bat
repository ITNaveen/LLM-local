@echo off
REM Live Translator - setup on Windows (for running it on the company laptop later).
REM Needs Python 3.10-3.12 from python.org ("py" launcher) and Ollama from https://ollama.com
cd /d "%~dp0"
py -3.12 -m venv .venv 2>nul || py -3 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip wheel
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m livetranslator download
echo.
echo Done. Start with start.bat
pause
