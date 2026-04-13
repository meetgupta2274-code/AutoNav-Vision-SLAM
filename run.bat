@echo off
title Visual SLAM Launcher
echo ===================================================
echo   Visual SLAM Autonomous Navigation System
echo ===================================================
echo.

echo [1/2] Booting up AI Backend Engine (FastAPI)...
start "AI Backend" cmd /k "cd backend && ..\myenv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak > nul

echo [2/2] Starting Telemetry Dashboard (React/Vite)...
start "Telemetry Dashboard" cmd /k "cd frontend && npm run dev"

echo.
echo All systems initialized! New terminal windows have been opened for the services.
echo.
echo Waiting for the dashboard to compile...
timeout /t 4 /nobreak > nul

echo.
echo ===================================================
echo SYSTEM READY: You can view the dashboard by opening
echo  -^> http://localhost:5173 (or localhost:5174)
echo ===================================================
pause
