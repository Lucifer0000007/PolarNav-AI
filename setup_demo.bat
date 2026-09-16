@echo off
REM PolarNav AI - one-time offline-prep install. Run this BEFORE demo day,
REM while you still have internet. Installs torch (CPU-only wheel) plus
REM everything in requirements.txt, so demo day itself needs zero pip and
REM zero network. System Python, no venv.
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
echo Done. demo.bat now runs fully offline.
