@echo off
REM Starts the MyaScreen backend on http://127.0.0.1:5000
cd /d "%~dp0backend\mg_predictor"
python -m pip install -r requirements.txt
python api.py
pause
