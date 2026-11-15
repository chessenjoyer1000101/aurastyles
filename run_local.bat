@echo off
rem Aura Styles - run on this PC only:  http://localhost:8000
cd /d "%~dp0"
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
