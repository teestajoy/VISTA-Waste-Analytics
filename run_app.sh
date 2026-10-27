#!/usr/bin/env bash
# Linux/macOS: ./run_app.sh      Windows PowerShell: $env:PYTHONPATH="src"; python -m uvicorn vista.app.api:app --port 8000
export PYTHONPATH=src
python -m uvicorn vista.app.api:app --host 127.0.0.1 --port 8000
