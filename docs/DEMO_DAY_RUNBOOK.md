# PolarNav AI — Demo Day Runbook (SIH26059)

## Boot sequence

1. **Before demo day, with internet:** run `setup_demo.bat` once. Installs the pinned CPU torch wheel plus everything in `requirements.txt`. Confirm it finishes without error.
2. **On demo day:** turn Wi-Fi off, then run `demo.bat`. System Python, no venv — this is the same launcher used in every QA pass this session.
3. Browser opens to `http://localhost:8501`. If it doesn't auto-open, navigate there manually.

## Wi-Fi-off verification (do this once before judges arrive)

1. Confirm Wi-Fi is actually off (not just "app looks fine" — check the OS network indicator).
2. Open browser DevTools → Network tab before clicking anything.
3. Click through all three buttons once (data drop, detect ice, compute route) and open the Strict JSON expander.
4. Confirm the Network tab shows **only `localhost` requests** — nothing else. This has been verified this session with every non-localhost request actively blocked at the network layer: zero external requests fired through the full flow.
5. Close and restart the browser tab before judges arrive, so the click sequence below starts from a clean load.

## Click order, with one spoken line each

| # | Click | Spoken line |
|---|---|---|
| 1 | **Simulate Satellite Data Drop** | "This ingests our iceberg positions, wind and current data, and generates today's SAR patch — scaled to match a real historical Antarctic ice extent from NSIDC, shown right here with the exact date and extent." |
| 2 | **Detect Ice (U-Net / OpenCV Surrogate)** | "This is the segmentation step. Otsu thresholding runs by default — the caption tells you honestly which model is active, and it flips automatically the moment trained weights clear our accuracy bar, no code change needed." |
| 3 | **Compute Risk-Aware Route** | "Here's the core: a 40-by-40 risk grid, advected 24 hours forward by wind and current, routed with a modified A* that treats ice as a cost penalty. Green is our optimized route, red is the naive direct line." |
| 4 | *(point at the map overlay)* | "The blue shading is our 24-hour ice concentration forecast. The route is planned against where the ice will be tomorrow, not just where it sits right now." |
| 5 | *(point at Live Alerts panel)* | "Every nearby iceberg gets a closest-point-of-approach calculation, colour-coded by severity. This is display-only — the captain always keeps final authority over the route." |
| — | *(optional: expand Strict JSON)* | "And this is the exact payload a shipboard system would receive — a vessel-API-ready schema. Transport into an actual shipboard system is Phase 2; today we're proving the payload itself." |

## If the live demo fails

- Have the backup recording ready and cued to timestamp 0:00 before judges sit down, not searched for live.
- One line to say while it loads: "We've hit a snag — here's a recording of the exact same run from rehearsal so you don't lose the thread."
- Do not attempt to debug live in front of judges. Switch to the recording, finish the narrative, offer to show the live app again afterward if time allows.

## The 4 Q&A shields

**1. "Your accuracy numbers seem low / where's your accuracy metric?"**
Pixel accuracy is a trap on ice masks: a frame that's 90% open water scores 90% accuracy by predicting "all water" and misses every iceberg. We report Dice and IoU instead, which actually measure overlap on the ice itself. The reference notebook's own 8-class baseline on a different, real-photo dataset scores IoU 0.441 — a useful comparison point, not our own number. Our binary SAR model hasn't been trained yet (zero labelled patches on this machine), so we're not claiming a number we don't have. We're gated on Dice ≥0.60 and beating Otsu by 0.05 before anything ships.

**2. "Your drift model's R² is basically 1.0 — that seems too good."**
It is too good, on purpose, and we say so in the report. The Ridge model is trained on synthetic samples generated *from* the physics formula plus noise, because no observed iceberg-displacement dataset exists yet. An R² near 1.0 means it correctly learned to reproduce the formula it was shown — that's a pipeline certification (train → serialize → load → infer → fallback all work), not a claim of forecasting skill on real-world drift. The physics formula itself remains the permanent, always-reachable fallback.

**3. "Why does your optimized route cost 38% more distance?"**
That's the safety-for-distance trade-off, and it's a tunable knob, not a fixed cost. The A* risk penalty weight controls how aggressively the route avoids ice; today's default buys a 71.7% cut in modeled ice exposure for a 38.2% longer route. A vessel with different risk tolerance turns that one number up or down — we're not hard-coding a single answer, we're demonstrating the trade-off exists and is controllable.

**4. "Is that a real satellite image?"**
No, and we don't claim it is — the caption says "synthetic sample." What's real is the *density*: every generated patch is scaled by an actual historical Antarctic sea-ice extent value drawn from the NSIDC Sea Ice Index (1978–2019, over 13,000 daily records), and the badge on screen shows the exact date and extent it drew. Drop a real Sentinel-1 crop into `data/sar_real.png` and the app uses that instead automatically — the surrogate exists because we don't have a real crop on this machine today, not because the pipeline can't take one.
