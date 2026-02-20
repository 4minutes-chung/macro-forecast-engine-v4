# Macro Risk V4

## 1) What this package is
Macro Risk V4 is the current governed quarterly macro scenario package for downstream PD and risk workflows.
It generates macro forecast paths, short-horizon uncertainty intervals, validation evidence, and PD-ready handoff files.

Validation contract in this package: `v4.1` (including V4.2 guardrail retune settings in config/tests).

## 2) Status (from `validation_summary.json`)
Source: `outputs/macro_engine/validation/validation_summary.json`

Current status:
- Release gate (incumbent, `champion_a`): `PASS`
- Release gate (challenger, `champion_b`): `PASS`
- Promotion gate: `PASS`
- Diagnostic-only mode: `false`

Current key metrics:
- Incumbent coverage pass-rate: `0.9444`
- Challenger coverage pass-rate: `1.0000`
- Incumbent width ratio mean: `1.3438`
- Challenger width ratio mean: `1.2748`
- Incumbent median rRMSE h1..h2: `0.9744`
- Incumbent median rRMSE h3..h4: `0.9208`
- Incumbent mean CRPS gain h5..h12 vs RW: `15.1304%`
- Challenger CRPS gain h9..h12 vs incumbent: `14.1399%`
- Challenger short-horizon CRPS worsen h1..h4: `-8.6351%` (improved, not worsened)

Interpretation:
- V4 is publishable as the current governed package.
- V3 remains available as rollback/fallback baseline.

## 3) Canonical handoff files
Use these three files for PD ingestion:
- `outputs/macro_engine/pd_regressors_forecast_levels.csv` (canonical)
- `outputs/macro_engine/pd_regressors_forecast_derived.csv` (convenience)
- `outputs/macro_engine/pd_regressors_metadata.json` (contract metadata)

Primary level targets in canonical handoff:
- `unemployment_rate`
- `ust10_rate`
- `hpi_yoy` (mapped from `hpi_growth_yoy`)

## 4) Repository map
Top-level:
- `macro_engine_config.json`: assumptions, threshold profiles, calibration controls, and validation contract
- `macro_engine_plain_report_v4.tex` / `macro_engine_plain_report_v4.pdf`: technical report and release record
- `macro_engine_schema.md`: output schema and artifact map
- `run_all.sh`: end-to-end pipeline runner
- `README.md`: this overview

Data:
- `data/macro_panel_quarterly_raw.csv`: transformed quarterly panel before modeling subset filtering
- `data/macro_panel_quarterly_model.csv`: modeling panel used by forecast/validation scripts
- `data/macro_panel_metadata.json`: source and transformation metadata

Scripts:
- `scripts/fetch_macro_panel_fred.py`: data refresh and transformation
- `scripts/run_macro_forecast_engine.py`: forecast generation
- `scripts/run_macro_validation.py`: validation gates, calibration, promotion checks, and artifacts
- `scripts/backtest_bvar_oos.py`: OOS backtest mechanics
- `scripts/export_pd_macro_subset.py`: PD export generation
- `scripts/macro_model_core.py`: reusable model/scoring utilities

Outputs:
- `outputs/macro_engine/`: forecast outputs and champion maps
- `outputs/macro_engine/validation/`: validation summary, calibration factors, and audit diagnostics

Tests:
- `tests/test_v42_validation_guards.py`: guardrail unit checks (tie-break behavior, calibration overrides, monotonicity, boundary consistency)

## 5) Model + governance summary
Model structure:
- Forecast horizon: 80 quarters
- Incumbent regime (`champion_a`): Q1..Q12 short, Q13..Q24 bridge, Q25..Q80 long-run/scenario
- Challenger regime (`champion_b`): Q1..Q16 short, Q17..Q28 bridge, Q29..Q80 long-run/scenario

Short-horizon champion selection:
- Candidate models: `BVAR`, `AR`, `RW`
- Selection by variable and bucket (`Q1..Q4`, `Q5..Q12`)
- Tie-break: if CRPS gap is small, use coverage+width balance score (`coverage_weight`, `width_weight`)

Calibration:
- Bucket-level interval scaling to target `coverage90 = 0.90`
- Width penalty to avoid uncontrolled interval expansion
- V4.2 challenger guardrail for `ust10_rate` bucket `5..12`:
  - tighter `scale_max`
  - tighter `width_cap`
  - higher width penalty

Validation/promotion governance:
- Power gate: required-cell `n_oos >= 40`
- Release profile checks: rRMSE (h1..h4), CRPS gain (h5..h12), coverage pass-rate, width discipline
- Hard checks: boundary consistency and scenario timing/ordering
- Promotion checks: challenger gain in h9..h12, short-horizon non-worsening, boundary comparator, and both release passes

## 6) What changed from V3 to V4
V3 was the first frozen governance baseline, but challenger did not promote.
V4 focused on closing that gap without breaking release discipline.

Key changes:
- Challenger evaluation moved from proxy-only path to challenger-specific champion map and calibration artifacts
- Tie-break behavior made explicit with coverage/width weighted balance
- Calibration controls became tighter and bucket-aware for weak cells
- V4.2 guardrails added for challenger `ust10_rate` medium bucket behavior
- Guardrail unit tests added and locked under `tests/test_v42_validation_guards.py`

Outcome shift (V3 -> V4):
- Challenger release: `FAIL` -> `PASS`
- Promotion: `FAIL` -> `PASS`
- Challenger h9..h12 CRPS gain vs incumbent: `1.6426%` -> `14.1399%`
- Challenger width ratio mean: `1.5285` -> `1.2748`

## 7) Residual risks and controls
Residual risks still visible:
- `coverage_fail_cells.csv` currently has 2 rows, both `hpi_growth_yoy` in bucket `Q5..Q12`:
  - `h=6` coverage `0.7727`, width ratio `1.4147`
  - `h=11` coverage `0.7955`, width ratio `1.7148`

Controls in place:
- Release and promotion gates are still passing
- Boundary consistency and scenario checks are passing
- V3 fallback package is retained for rollback
- Legacy sample export is opt-in only (`WRITE_LEGACY_SAMPLE=1`)

## 8) Repro commands
Run step-by-step:
```bash
python3 scripts/fetch_macro_panel_fred.py \
  --raw-output data/macro_panel_quarterly_raw.csv \
  --model-output data/macro_panel_quarterly_model.csv \
  --metadata-output data/macro_panel_metadata.json

python3 scripts/run_macro_forecast_engine.py \
  --config macro_engine_config.json \
  --output-dir outputs/macro_engine

python3 scripts/run_macro_validation.py \
  --config macro_engine_config.json \
  --input data/macro_panel_quarterly_model.csv \
  --output-dir outputs/macro_engine/validation \
  --champion-map-output outputs/macro_engine/champion_map.json \
  --verbose-validation

python3 scripts/export_pd_macro_subset.py \
  --config macro_engine_config.json \
  --input outputs/macro_engine/macro_forecast_paths.csv \
  --model-panel data/macro_panel_quarterly_model.csv \
  --levels-output outputs/macro_engine/pd_regressors_forecast_levels.csv \
  --derived-output outputs/macro_engine/pd_regressors_forecast_derived.csv \
  --metadata-output outputs/macro_engine/pd_regressors_metadata.json
```

Or run end-to-end:
```bash
WRITE_LEGACY_SAMPLE=0 ./run_all.sh
```

Legacy sample mode (only if needed):
```bash
WRITE_LEGACY_SAMPLE=1 ./run_all.sh
```

## 9) References
- Technical report: `macro_engine_plain_report_v4.pdf`
- Schema and field definitions: `macro_engine_schema.md`
- Validation evolution record: `VALIDATION_ENGINE_FROM_V1.md`
