@echo off
rem Aura Styles - start the app and open a temporary public link for friends.
rem Needs cloudflared (free, no account):  winget install Cloudflare.cloudflared
rem The link works only while this window stays open and this PC is on.
cd /d "%~dp0"
where cloudflared >nul 2>nul
if errorlevel 1 (
  echo cloudflared is not installed. Run:  winget install Cloudflare.cloudflared
  echo then open a new terminal and start this file again.
  pause
  exit /b 1
)
start "Aura Styles server" cmd /k python -m uvicorn app.server:app --host 127.0.0.1 --port 8000
echo Waiting for the model to load...
timeout /t 25 /nobreak >nul
echo.
echo Your public link is the https://....trycloudflare.com address printed below.
echo.
cloudflared tunnel --url http://127.0.0.1:8000
