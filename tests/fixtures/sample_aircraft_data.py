"""
Sample Aircraft Data Fixtures for Unit and Integration Testing

Reference: specs/001-aircraft-range-prediction/data-model.md
"""

import pandas as pd
from typing import Tuple


def get_sample_aircraft_data(n_samples: int = 10) -> pd.DataFrame:
    """
    Get small representative aircraft dataset for unit/integration testing.

    Args:
        n_samples: Number of aircraft samples to return

    Returns:
        DataFrame with schema: aircraft_id, passenger_capacity, num_engines,
        crew_size, wingspace_m, length_m, range_km
    """
    data = {
        "aircraft_id": [f"AC{i:03d}" for i in range(1, n_samples + 1)],
        "passenger_capacity": [150, 250, 180, 320, 100, 200, 280, 140, 360, 110],
        "num_engines": [2, 4, 2, 4, 2, 3, 4, 2, 4, 2],
        "crew_size": [6, 8, 5, 9, 4, 6, 8, 5, 10, 4],
        "wingspace_m": [35.8, 64.4, 33.5, 65.0, 28.4, 40.0, 62.0, 32.0, 68.0, 29.0],
        "length_m": [70.7, 70.7, 63.7, 73.8, 55.3, 65.0, 72.0, 60.0, 76.0, 57.0],
        "range_km": [12500, 14000, 11000, 14500, 8500, 11500, 13500, 9800, 15000, 8000],
    }
    return pd.DataFrame(data).head(n_samples)


def get_sample_aircraft_data_with_missing_values() -> pd.DataFrame:
    """
    Get aircraft data with missing values for testing imputation.

    Returns:
        DataFrame with NaN values in some feature columns
    """
    df = get_sample_aircraft_data(5)
    df.loc[0, "crew_size"] = None  # Missing value
    df.loc[2, "wingspace_m"] = None  # Missing value
    return df


def get_sample_aircraft_data_with_outliers() -> pd.DataFrame:
    """
    Get aircraft data with outliers for testing outlier removal.

    Returns:
        DataFrame with extreme outlier values
    """
    df = get_sample_aircraft_data(5)
    df.loc[1, "passenger_capacity"] = 9999  # Extreme outlier
    df.loc[3, "range_km"] = 100  # Below realistic range
    return df


def get_sample_preprocessed_features() -> pd.DataFrame:
    """
    Get normalized aircraft features (after StandardScaler) for testing.

    Returns:
        DataFrame with normalized columns: [feature]_normalized, plus
        aircraft_id, range_km, split
    """
    data = {
        "aircraft_id": ["AC001", "AC002", "AC003", "AC004", "AC005"],
        "passenger_capacity_normalized": [-0.5, 1.2, 0.3, 1.8, -1.2],
        "num_engines_normalized": [-0.7, 1.5, -0.7, 1.5, -0.7],
        "crew_size_normalized": [-0.4, 1.1, -0.1, 1.4, -0.8],
        "wingspace_m_normalized": [-0.6, 1.3, -0.8, 1.4, -1.2],
        "length_m_normalized": [0.2, 0.2, -0.5, 0.6, -1.1],
        "range_km": [12500, 14000, 11000, 14500, 8500],
        "split": ["train", "train", "val", "val", "test"],
    }
    return pd.DataFrame(data)


def get_sample_predictions() -> pd.DataFrame:
    """
    Get sample predictions from MC Dropout inference.

    Returns:
        DataFrame with schema: aircraft_id, predicted_range_km,
        range_lower_ci_km, range_upper_ci_km, prediction_confidence
    """
    data = {
        "aircraft_id": ["AC001", "AC002", "AC003", "AC004", "AC005"],
        "predicted_range_km": [12400, 13950, 10900, 14550, 8450],
        "range_lower_ci_km": [12100, 13600, 10500, 14200, 8100],
        "range_upper_ci_km": [12700, 14300, 11300, 14900, 8800],
        "prediction_confidence": [0.85, 0.90, 0.78, 0.92, 0.75],
    }
    return pd.DataFrame(data)


def get_sample_evaluation_metrics() -> dict:
    """
    Get sample evaluation metrics for testing.

    Returns:
        Dict with RMSE, R², MAE metrics overall and by aircraft type
    """
    return {
        "rmse_overall": 450.5,
        "r2_overall": 0.87,
        "mae_overall": 320.2,
        "rmse_by_type": {
            "twin_engine": 380.0,
            "four_engine": 520.0,
        },
        "r2_by_type": {
            "twin_engine": 0.90,
            "four_engine": 0.84,
        },
        "mae_by_type": {
            "twin_engine": 280.0,
            "four_engine": 360.0,
        },
        "ci_calibration": 0.92,  # 92% of actual values in predicted CI
        "drift_flag": False,  # No significant model drift detected
    }
