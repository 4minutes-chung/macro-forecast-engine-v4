# Validation Engine From V1

Last updated: 2026-02-20
Package focus: `/Users/stevenchung/Desktop/P12B_File/Macro risk v4`

## 1) Purpose
This file is the validation history and current gate dashboard from V1 to V4.2.
It replaces older scorecard + bootstrap validation notes.

## 2) Evolution by version
| Version | Validation maturity | Gate snapshot | Key advancement |
|---|---|---|---|
| V1 | Prototype outputs + backtest table only | No formal `validation_summary.json` gate package | Established quarterly base pipeline |
| V2/V2.1 | Checklist-style monthly validation docs | No quarterly release/promotion gate contract | Added monthly branch and panel-stability fix |
| V3 | Formal quarterly machine-readable gate artifacts | Inc release PASS, challenger release FAIL, promotion FAIL | First true governance baseline |
| V4/V4.2 | Extended gate logic + guardrails + micro-sprint no-regression checks | Inc release PASS, challenger release PASS, promotion PASS | Promotion unlocked with bounded residual risk |

## 3) Current V4.2 gate snapshot
Source:
- `outputs/macro_engine/validation/validation_summary.json`

### Release profile
| Metric | Threshold | Incumbent | Challenger | Status |
|---|---:|---:|---:|---|
| Release pass | required | True | True | PASS |
| Min required-cell n_oos | >= 40 | 44 | 40 | PASS |
| Coverage90 pass-rate | >= 0.75 | 0.9444 | 1.0000 | PASS |
| Width ratio mean | <= 1.35 | 1.3438 | 1.2748 | PASS |
| Width ratio per-var max | <= 1.60 | 1.4779 | 1.3767 | PASS |
| Boundary consistency | pass | True | True | PASS |
| Scenario pass | pass | True | True | PASS |

### Promotion profile
| Metric | Threshold | Actual | Status |
|---|---:|---:|---|
| Promotion pass | pass | True | PASS |
| CRPS gain h9..h12 (challenger vs incumbent) | >= 5.0% | 14.1399% | PASS |
| Short-horizon CRPS worsen h1..h4 | <= 1.0% | -8.6351% | PASS |
| Boundary comparator pass | required | True | PASS |

## 4) Residual diagnostics
Sources:
- `outputs/macro_engine/validation/micro_sprint_day1_baseline.json`
- `outputs/macro_engine/validation/coverage_fail_cells.csv`

- Baseline coverage-fail cells: `5`
- Current coverage-fail cells: `2`
- Remaining:
  - `hpi_growth_yoy`, h=6, bucket `5..12`
  - `hpi_growth_yoy`, h=11, bucket `5..12`

## 5) What changed in V4 validation logic
1. Challenger evaluated with challenger champion outputs (not proxy path).
2. Regime-aware scenario timing check.
3. Small-gap tie-break prefers coverage+width balance.
4. Challenger ust10 long-bucket calibration controls.
5. Calibration monotonicity check.
6. Boundary consistency check in gate evaluation.

## 6) Decision
Validation decision for V4 package state: `APPROVE FOR CLOSURE`.

Conditions satisfied:
1. Release pass for incumbent and challenger.
2. Promotion pass.
3. No-regression criterion met (coverage-fail count reduced 5 -> 2).
4. Scenario and boundary checks pass.

## 7) Reproduce validation snapshot
```bash
python3 -m unittest -v tests/test_v42_validation_guards.py

python3 scripts/run_macro_validation.py \
  --config macro_engine_config.json \
  --input data/macro_panel_quarterly_model.csv \
  --output-dir outputs/macro_engine/validation \
  --champion-map-output outputs/macro_engine/champion_map.json \
  --verbose-validation
```
