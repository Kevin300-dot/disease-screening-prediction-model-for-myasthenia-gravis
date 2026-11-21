@echo off
echo ========================================================
echo   Starting MyaScreen (Backend + Frontend)
echo ========================================================

echo [1/2] Starting Python backend on http://127.0.0.1:5000 ...
start "MyaScreen Backend" cmd /k "cd /d "%~dp0backend\mg_predictor" && python api.py"

echo [2/2] Starting Vite Frontend...
start "MyaScreen Frontend" cmd /k "cd /d "%~dp0" && npm run dev"

echo.
echo Both servers have been launched in separate windows!
echo Backend API : http://127.0.0.1:5000
echo Web App     : Check the Frontend terminal window for the local port (e.g., http://localhost:8080 or http://localhost:3000)
echo.
pause
