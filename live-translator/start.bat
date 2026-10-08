@echo off
REM Live Translator - start on Windows. Extra arguments are passed on, e.g.:
REM   start.bat --host 0.0.0.0 --token SECRET
cd /d "%~dp0"
.venv\Scripts\python -m livetranslator serve --open %*
