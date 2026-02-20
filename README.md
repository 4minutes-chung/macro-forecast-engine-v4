# Macro Risk V4

## Overview
Macro Risk V4 is the current quarterly macro scenario engine in this project line.
It produces forecast paths, uncertainty intervals, and validation evidence used by downstream risk workflows.

V4 is the first version where all governance decisions pass in one snapshot:
- incumbent release: `PASS`
- challenger release: `PASS`
- promotion: `PASS`

Validation contract: `v4.1` (with V4.2 guardrail enhancements).

## Why V4
V3 established a stable governance baseline but did not clear promotion.
V4 focused on closing that gap through targeted validation and calibration improvements while controlling regression risk.

## Current Validation Snapshot
Source: `outputs/macro_engine/validation/validation_summary.json`

Release profile:
- incumbent coverage pass-rate: `0.9444`
- challenger coverage pass-rate: `1.0000`
- incumbent width ratio mean: `1.3438`
- challenger width ratio mean: `1.2748`
- incumbent median rRMSE h1..h2: `0.9744`
- incumbent median rRMSE h3..h4: `0.9208`
- incumbent mean CRPS gain h5..h12 vs RW: `15.1304%`

Promotion profile:
- CRPS gain h9..h12 (challenger vs incumbent): `14.1399%`
- short-horizon CRPS worsen h1..h4: `-8.6351%` (improved)

Residual risk:
- 2 remaining coverage-fail cells, both in `hpi_growth_yoy` (h=6, h=11, bucket `5..12`)

## Core Deliverables
Key deliverables from this package:
- `outputs/macro_engine/macro_forecast_paths.csv`
- `outputs/macro_engine/macro_forecast_short_horizon_intervals.csv`
- `outputs/macro_engine/champion_map.json`
- `outputs/macro_engine/champion_map_champion_b.json`
- `outputs/macro_engine/validation/validation_summary.json`
- `outputs/macro_engine/validation/calibration_factors.json`
- `outputs/macro_engine/validation/coverage_fail_cells.csv`

Handoff set:
- `outputs/macro_engine/pd_regressors_forecast_levels.csv`
- `outputs/macro_engine/pd_regressors_forecast_derived.csv`
- `outputs/macro_engine/pd_regressors_metadata.json`

## Repository Layout
- `macro_engine_config.json`: assumptions, thresholds, and validation settings
- `data/`: quarterly panel and metadata
- `scripts/`: data, forecast, validation, and export logic
- `outputs/macro_engine/`: model outputs and validation artifacts
- `tests/`: validation guardrail checks

## Documentation
- technical report: `macro_engine_plain_report_v4.pdf`
- report source: `macro_engine_plain_report_v4.tex`
- schema and field definitions: `macro_engine_schema.md`
- validation evolution (V1 -> V4): `VALIDATION_ENGINE_FROM_V1.md`

## Reproduce
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
bash run_all.sh
```

## Next Steps
V4 is closed for a controlled release state:
- gates pass,
- residual risk is documented,
- V3 remains available as rollback benchmark.
