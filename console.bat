@echo off
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate
python -m uvicorn api:app --host 127.0.0.1 --port 8000
