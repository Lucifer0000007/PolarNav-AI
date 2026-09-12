@echo off
cd /d "%~dp0"
if exist venv\Scripts\activate.bat call venv\Scripts\activate
python -m streamlit run app.py
