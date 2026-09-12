# PolarNav AI — SCOPE LOCK
Streamlit offline Antarctic navigation demo (SIH26059). 
Editable files ONLY: engine.py, app.py, .streamlit/**, demo.bat, data/**.
NEVER edit: README.md, PPT files, anything outside this folder.

## Feature contract F1–F9 (the only features that exist)
F1 data-drop · F2 OpenCV mask+count · F3 risk grid · F4 24h drift arrows ·
F5 A* green + direct red · F6 live metrics · F7 SQLite history ·
F8 offline no-keys · F9 strict JSON expander.

## Hard constraints
- No new dependencies. No new features/buttons/pages. No refactors/renames.
- No network calls, API keys, cloud/Gemini paths.
- Minimal diffs: 3-line fix over rewrite, always.
- Keep fallbacks (synthetic SAR, default CSVs, astar None fallback).
- After every edit run: venv\Scripts\python.exe engine.py (must pass).

## Current mission (do in order, nothing beyond)
1 real-SAR wiring · 2 offline map · 3 config.toml · 4 lat/lon labels ·
5 scenario tuning · 6 demo.bat · then /qa.
