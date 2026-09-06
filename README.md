# PolarNav AI — Offline Antarctic Navigation Decision Support
SIH26059 · TEKATHON-5.0 (2026) · Team Zero Output
Offline-first decision support for NCPOR vessels: SAR ice detection, 24-h
iceberg drift prediction, risk-aware A* routing — zero internet required.

## Features
- Simulated satellite data drops (SAR + wind/current + iceberg coords)
- OpenCV ice-hazard detection (blur → Otsu → morphology)
- 24-h drift prediction (vector kinematics: current + 3% wind)
- Risk-aware Modified A* routing, f(n) = g(n) + h(n) + p(n)
- Direct vs optimized route comparison with live metrics
- SQLite local route history (persists offline)
- Strict JSON vessel-API output
- Runs fully offline — no cloud APIs, no keys

## Install
python -m venv venv
venv\Scripts\activate
pip install streamlit folium streamlit-folium pandas numpy opencv-python

## Run
python engine.py        # offline self-test
streamlit run app.py    # dashboard → http://localhost:8501

## Demo flow
1. Simulate Satellite Data Drop → 2. Detect Ice Hazards →
3. Predict Drift + Generate Route → check history + strict JSON.