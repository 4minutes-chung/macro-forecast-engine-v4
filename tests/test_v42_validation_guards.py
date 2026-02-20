#!/usr/bin/env python3
"""Unit checks for V4.2 validation guardrails."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_macro_validation as rmv  # noqa: E402


class ValidationGuardsTest(unittest.TestCase):
    def test_calibration_coverage_monotonicity(self) -> None:
        abs_err = np.array([0.05, 0.10, 0.20, 0.40, 0.80], dtype=float)
        half_width = np.array([0.08, 0.08, 0.08, 0.08, 0.08], dtype=float)
        self.assertTrue(
            rmv.calibration_coverage_monotonicity(
                abs_err=abs_err,
                half_width=half_width,
                scale_min=0.8,
                scale_max=1.6,
                grid_points=31,
            )
        )

    def test_boundary_stats_consistency(self) -> None:
        ok = {
            "combined": {"max_z": 0.5, "median_z": 0.2, "n": 6},
            "short": {"max_z": 0.4, "median_z": 0.1, "n": 3},
            "medium": {"max_z": 0.5, "median_z": 0.2, "n": 3},
        }
        bad = {
            "combined": {"max_z": 0.2, "median_z": 0.3, "n": 6},
            "short": {"max_z": 0.2, "median_z": 0.1, "n": 2},
            "medium": {"max_z": 0.2, "median_z": 0.1, "n": 2},
        }
        self.assertTrue(rmv.boundary_stats_consistency(ok))
        self.assertFalse(rmv.boundary_stats_consistency(bad))

    def test_tie_break_prefers_coverage_width_balance_when_gap_small(self) -> None:
        rows = []
        for h in range(1, 5):
            rows.extend(
                [
                    {
                        "regime_id": "champion_b",
                        "model": "BVAR",
                        "variable": "ust10_rate",
                        "horizon_q": h,
                        "crps": 0.5,
                        "coverage_90": 0.90,
                        "width_ratio_vs_rw": 1.1,
                    },
                    {
                        "regime_id": "champion_b",
                        "model": "AR",
                        "variable": "ust10_rate",
                        "horizon_q": h,
                        "crps": 0.7,
                        "coverage_90": 0.85,
                        "width_ratio_vs_rw": 1.0,
                    },
                    {
                        "regime_id": "champion_b",
                        "model": "RW",
                        "variable": "ust10_rate",
                        "horizon_q": h,
                        "crps": 0.8,
                        "coverage_90": 0.80,
                        "width_ratio_vs_rw": 1.0,
                    },
                ]
            )

        # In bucket_5_12, BVAR has slightly better CRPS but materially worse width balance.
        for h in range(5, 13):
            rows.extend(
                [
                    {
                        "regime_id": "champion_b",
                        "model": "BVAR",
                        "variable": "ust10_rate",
                        "horizon_q": h,
                        "crps": 1.00,
                        "coverage_90": 0.905,
                        "width_ratio_vs_rw": 1.60,
                    },
                    {
                        "regime_id": "champion_b",
                        "model": "AR",
                        "variable": "ust10_rate",
                        "horizon_q": h,
                        "crps": 1.01,
                        "coverage_90": 0.890,
                        "width_ratio_vs_rw": 1.05,
                    },
                    {
                        "regime_id": "champion_b",
                        "model": "RW",
                        "variable": "ust10_rate",
                        "horizon_q": h,
                        "crps": 1.20,
                        "coverage_90": 0.75,
                        "width_ratio_vs_rw": 1.0,
                    },
                ]
            )

        agg_df = pd.DataFrame(rows)
        config = {
            "model": {"variables": ["ust10_rate"]},
            "validation": {
                "ensemble_gap_pct": 1.5,
                "pd_calibration": {"target_coverage90": 0.90},
                "tie_break": {"coverage_weight": 1.0, "width_weight": 0.35},
            },
        }
        champ = rmv.build_champion_map(agg_df, config, "champion_b")
        self.assertEqual(champ["variables"]["ust10_rate"]["bucket_5_12"]["model"], "AR")

    def test_challenger_ust10_bucket_override_is_applied(self) -> None:
        rows = []
        rw_rows = []
        for origin in range(20):
            for h in range(5, 13):
                rows.append(
                    {
                        "regime_id": "champion_b",
                        "model": "CHAMPION",
                        "variable": "ust10_rate",
                        "horizon_q": h,
                        "origin_idx": origin,
                        "actual": 3.0 + 0.03 * h,
                        "pred": 3.0 + 0.02 * h,
                        "width90": 1.8,
                    }
                )
                rw_rows.append(
                    {
                        "regime_id": "champion_b",
                        "model": "RW",
                        "variable": "ust10_rate",
                        "horizon_q": h,
                        "origin_idx": origin,
                        "width90": 1.0,
                    }
                )

        config = {
            "validation": {
                "objective_mode_default": "pd_levels_primary",
                "pd_level_targets": ["ust10_rate"],
                "pd_source_mapping": {},
                "pd_calibration": {
                    "enabled": True,
                    "target_coverage90": 0.90,
                    "apply_to_buckets": ["bucket_5_12"],
                    "min_samples_per_bucket": 10,
                    "scale_min": 0.8,
                    "scale_max": 1.8,
                    "objective_lambda_width_penalty": 2.0,
                    "scale_grid_points": 21,
                },
                "calibration": {"target_coverage90": 0.90},
            }
        }

        factors = rmv.fit_calibration_factors_from_raw(
            raw_df=pd.DataFrame(rows),
            model_name="CHAMPION",
            config=config,
            regime_id="champion_b",
            rw_reference_df=pd.DataFrame(rw_rows),
            calibration_context="challenger",
        )
        f = factors["ust10_rate"]["bucket_5_12"]
        self.assertTrue(f["challenger_override_applied"])
        self.assertEqual(f["calibration_context"], "challenger")
        self.assertLessEqual(float(f["width_cap"]), 1.35 + 1e-9)

    def test_per_target_bucket_override_can_enable_bucket_not_globally_applied(self) -> None:
        rows = []
        rw_rows = []
        for origin in range(20):
            for h in range(1, 5):
                rows.append(
                    {
                        "regime_id": "champion_a",
                        "model": "CHAMPION",
                        "variable": "ust10_rate",
                        "horizon_q": h,
                        "origin_idx": origin,
                        "actual": 3.0 + 0.03 * h,
                        "pred": 3.0 + 0.02 * h,
                        "width90": 0.95,
                    }
                )
                rw_rows.append(
                    {
                        "regime_id": "champion_a",
                        "model": "RW",
                        "variable": "ust10_rate",
                        "horizon_q": h,
                        "origin_idx": origin,
                        "width90": 1.0,
                    }
                )

        config = {
            "validation": {
                "objective_mode_default": "pd_levels_primary",
                "pd_level_targets": ["ust10_rate"],
                "pd_source_mapping": {},
                "pd_calibration": {
                    "enabled": True,
                    "target_coverage90": 0.90,
                    "apply_to_buckets": ["bucket_5_12"],
                    "min_samples_per_bucket": 10,
                    "scale_min": 0.8,
                    "scale_max": 1.8,
                    "objective_lambda_width_penalty": 2.0,
                    "scale_grid_points": 21,
                    "per_target": {
                        "ust10_rate": {
                            "bucket_overrides": {
                                "bucket_1_4": {
                                    "enabled": True,
                                    "scale_min": 1.0,
                                    "scale_max": 1.12,
                                    "width_cap": 1.10,
                                }
                            }
                        }
                    },
                },
                "calibration": {"target_coverage90": 0.90},
            }
        }

        factors = rmv.fit_calibration_factors_from_raw(
            raw_df=pd.DataFrame(rows),
            model_name="CHAMPION",
            config=config,
            regime_id="champion_a",
            rw_reference_df=pd.DataFrame(rw_rows),
            calibration_context="incumbent",
        )
        f = factors["ust10_rate"]["bucket_1_4"]
        self.assertTrue(f["bucket_override_applied"])
        self.assertGreaterEqual(float(f["calibration_scale"]), 1.0 - 1e-9)
        self.assertLessEqual(float(f["calibration_scale"]), 1.12 + 1e-9)


if __name__ == "__main__":
    unittest.main()
