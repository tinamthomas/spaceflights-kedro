"""
Unit Tests for Data Science Pipeline: Preprocessing (Phase 2 Foundational)

Reference: specs/001-aircraft-range-prediction/spec.md
Constitution: Test-First Development (TDD) — Tests written BEFORE implementation
These tests are written in RED phase (should fail before implementation exists)
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class TestPreprocessFeatures:
    """Tests for preprocess_features node (T013-T016)"""

    def test_preprocess_features_normalizes_to_unit_variance(self):
        """
        T013: Verify that features are normalized to unit variance (σ≈1).

        After StandardScaler normalization:
        - Mean of each feature ≈ 0 (within floating point tolerance)
        - Standard deviation ≈ 1

        Expected: RED (failing) until preprocess_features implemented
        """
        # This test expects preprocess_features to exist and be importable
        from src.spaceflights.pipelines.data_science.nodes import preprocess_features

        # Sample data (5 features + target)
        data = pd.DataFrame({
            "aircraft_id": [f"AC{i:03d}" for i in range(1, 11)],
            "passenger_capacity": [150, 250, 180, 320, 100, 200, 280, 140, 360, 110],
            "num_engines": [2, 4, 2, 4, 2, 3, 4, 2, 4, 2],
            "crew_size": [6, 8, 5, 9, 4, 6, 8, 5, 10, 4],
            "wingspace_m": [35.8, 64.4, 33.5, 65.0, 28.4, 40.0, 62.0, 32.0, 68.0, 29.0],
            "length_m": [70.7, 70.7, 63.7, 73.8, 55.3, 65.0, 72.0, 60.0, 76.0, 57.0],
            "range_km": [12500, 14000, 11000, 14500, 8500, 11500, 13500, 9800, 15000, 8000],
        })

        processed_data, scaler = preprocess_features(data, random_seed=42)

        # Check normalized features have mean ≈ 0, std ≈ 1
        normalized_cols = [
            c for c in processed_data.columns
            if "_normalized" in c
        ]
        for col in normalized_cols:
            assert abs(processed_data[col].mean()) < 0.1, f"{col} mean != 0"
            assert abs(processed_data[col].std() - 1.0) < 0.1, f"{col} std != 1"

    def test_preprocess_features_handles_missing_values(self):
        """
        T014: Verify that missing values are imputed (median strategy).

        Expected: RED (failing) until preprocess_features handles imputation
        """
        from src.spaceflights.pipelines.data_science.nodes import preprocess_features

        data = pd.DataFrame({
            "aircraft_id": ["AC001", "AC002", "AC003"],
            "passenger_capacity": [150, None, 180],
            "num_engines": [2, 4, 2],
            "crew_size": [6, 8, None],
            "wingspace_m": [35.8, 64.4, 33.5],
            "length_m": [70.7, 70.7, 63.7],
            "range_km": [12500, 14000, 11000],
        })

        processed_data, _ = preprocess_features(data, random_seed=42)

        # Check no NaN values remain after preprocessing
        assert processed_data.isna().sum().sum() == 0, "NaN values remain after preprocessing"

    def test_preprocess_features_split_assignment(self):
        """
        T015: Verify that data is split into train/val/test (70%/15%/15%).

        Expected: RED (failing) until split assignment implemented
        """
        from src.spaceflights.pipelines.data_science.nodes import preprocess_features

        data = pd.DataFrame({
            "aircraft_id": [f"AC{i:03d}" for i in range(1, 101)],
            "passenger_capacity": np.random.randint(100, 400, 100),
            "num_engines": np.random.randint(2, 5, 100),
            "crew_size": np.random.randint(4, 10, 100),
            "wingspace_m": np.random.uniform(28, 70, 100),
            "length_m": np.random.uniform(55, 77, 100),
            "range_km": np.random.randint(8000, 15000, 100),
        })

        processed_data, _ = preprocess_features(data, random_seed=42)

        # Check split column exists
        assert "split" in processed_data.columns, "split column missing"

        # Check split distribution: ~70% train, ~15% val, ~15% test
        split_counts = processed_data["split"].value_counts()
        train_pct = split_counts.get("train", 0) / len(processed_data)
        val_pct = split_counts.get("val", 0) / len(processed_data)
        test_pct = split_counts.get("test", 0) / len(processed_data)

        assert 0.65 < train_pct < 0.75, f"train split {train_pct} not ~70%"
        assert 0.10 < val_pct < 0.20, f"val split {val_pct} not ~15%"
        assert 0.10 < test_pct < 0.20, f"test split {test_pct} not ~15%"

    def test_preprocess_features_deterministic_with_seed(self):
        """
        T016: Verify that preprocess_features is deterministic with fixed seed.

        Same seed → identical output (for reproducibility)

        Expected: RED (failing) until seed is used correctly
        """
        from src.spaceflights.pipelines.data_science.nodes import preprocess_features

        data = pd.DataFrame({
            "aircraft_id": [f"AC{i:03d}" for i in range(1, 21)],
            "passenger_capacity": np.random.randint(100, 400, 20),
            "num_engines": np.random.randint(2, 5, 20),
            "crew_size": np.random.randint(4, 10, 20),
            "wingspace_m": np.random.uniform(28, 70, 20),
            "length_m": np.random.uniform(55, 77, 20),
            "range_km": np.random.randint(8000, 15000, 20),
        })

        # Run preprocessing twice with same seed
        result1, _ = preprocess_features(data.copy(), random_seed=42)
        result2, _ = preprocess_features(data.copy(), random_seed=42)

        # Check split assignments are identical
        pd.testing.assert_series_equal(
            result1["split"],
            result2["split"],
            check_names=True,
            check_dtype=True,
        )
