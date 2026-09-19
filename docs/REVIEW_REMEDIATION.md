# PolarNav AI — Review Remediation Log (SIH26059)

Seven findings from a review pass against the shipped prototype, fixed and
verified 2026-09-19. One commit per finding, each independently revertible.
Source of truth for every number below: `python engine.py` / `python
engine.py --sanity`, run fresh at closeout, not carried over from memory.

| # | Finding | Fix applied | Commit | Verification |
|---|---|---|---|---|
| F1 | Streamlit usage-stats telemetry sent on every run | `[browser] gatherUsageStats = false` in `.streamlit/config.toml` | `6dfa7ed` | Flag present; blocked-hosts run (all non-localhost requests aborted) at startup + full click flow: zero external requests |
| F2 | Iceberg risk stamping used plain array assignment — two icebergs rounding to the same 40x40 cell let the later one silently overwrite the earlier one's higher risk | `np.maximum.at(risk, (rs, cs), weights)`, order-independent | `bf31fac` | New self-test block: both array orderings yield the correct max (10.0); plain assignment would have given 7.0 in the high-then-low case |
| F3 | `classify_threat()` forced HIGH whenever the route merely crossed a predicted risk>5 cell, regardless of how far any iceberg actually was | HIGH is now proximity-only (CPA<5km); crossings became an independent "MED: Route crosses N predicted risk>5 cells - expect icebreaking" advisory that never escalates to HIGH | `49aeb81` | Self-test assertion: pinned sample (CPA 14.22/23.18 km, both >=10, predicted_crossings=1) now reports LOW, not HIGH; live Playwright run shows both alert types rendering independently, 0 exceptions |
| F4 | KMeans clustered icebergs on raw (mass_kt, freeboard_m) — mass's much larger numeric scale drowned out freeboard entirely | Each feature divided by its own column max before fitting; tier order still ranked by raw mass specifically | `95a5ddc` | Verified on live `data/icebergs.csv`: 7 of 10 icebergs shift tier; the iceberg with the highest freeboard in the set (30.54m) no longer ties with one at 8.04m purely because their masses were similar |
| F5 | "Vessel API contract/integration" wording overclaimed a tested shipboard transport that doesn't exist | Reworded to "vessel-API-ready payload schema (transport = Phase 2)" in 4 places (app.py expander label, README.md table row/tree comment/heading); added `routes_out/latest.json` file-drop with a sha256 checksum field | `2ce59c4` | Grep for the overclaim pattern: zero hits; live check: export file exists with the sha256 key after one route compute |
| F6 | No upfront, hard-to-miss statement of model status before the reader hits any pipeline detail | New "Model status (read first)" section: `docs/JUDGE_GUIDE.md` (new), `README.md` (promoted from an existing footnote), `app.py` (one `st.info()` line under the title) | `37e82ad` | Strings present in all three; `JUDGE_GUIDE.pdf` re-rendered, confirmed still exactly 2 pages (964 words) |
| F7 | No automated check that the routing/risk pipeline generalizes beyond the one pinned demo scenario | `python engine.py --sanity`: 3 alternate (seed, start, goal) configs, asserting a path is found, every metric is finite, and optimized risk <= direct risk | `51053ff` | No-args path confirmed byte-identical to pre-change baseline (masking only the timestamp and DB-row-count lines, which vary run to run regardless); `--sanity` exits 0 |

## Post-remediation full-stack verification

- `/qa`: **GO** — F1-F9 flow, out-of-order clicks, refresh persistence, live metrics across two goals, all pass with 0 exceptions.
- `/security-review`: **clean** — zero findings; the only new I/O is one hardcoded local file write, no user-controlled path construction anywhere.
- `python engine.py --sanity`: **all 3 configs PASS** — seed 7 (start (8,8)→goal (32,30)): 7.7% risk reduction; seed 42 ((3,20)→(36,15)): 86.1%; seed 1337 ((10,2)→(25,38)): 96.9%.

## Cascade impact on demo numbers

Confirmed at closeout: **F2 and F4 have zero effect on the pinned self-test**
(it never passes real iceberg data into `build_risk_grid`). **F3 changes
exactly one self-test line** (`threat HIGH` → `threat LOW`) and touches
nothing else. The headline numbers — 71.7% risk reduction, +38.2% distance
cost — are identical before and after all seven fixes. `docs/JUDGE_GUIDE.md`'s
numbers footer required no update.
