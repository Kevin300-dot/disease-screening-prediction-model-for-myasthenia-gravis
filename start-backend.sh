#!/usr/bin/env bash
# Starts the MyaScreen backend on http://127.0.0.1:5000
cd "$(dirname "$0")/backend/mg_predictor" || exit 1
python3 -m pip install -r requirements.txt
python3 api.py
