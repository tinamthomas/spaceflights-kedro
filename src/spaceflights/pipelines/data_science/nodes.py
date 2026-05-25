"""
Kedro Pipeline for Aircraft Range Prediction: Data Science Nodes

All nodes are pure functions following Kedro best practices:
- Deterministic (no side effects; same input → same output)
- Testable (no global state)
- Documented (I/O schemas, type hints, docstrings)

Reference: specs/001-aircraft-range-prediction/plan.md
Constitution: Test-First Development, Kedro pure functions, data quality metadata
"""

import logging
from typing import Dict, Tuple, Any, List
import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


def validate_input_schema(
    data: pd.DataFrame,
    expected_columns: List[str],
    data_types: Dict[str, str],
) -> None:
    """
    Validate input data schema against specification.

    Args:
        data: Input DataFrame to validate
        expected_columns: List of required column names
        data_types: Dict mapping column name to expected dtype (e.g., {"age": "int64"})

    Returns:
        None

    Raises:
        ValueError: If schema validation fails

    Example:
        >>> validate_input_schema(
        ...     df,
        ...     expected_columns=["aircraft_id", "passenger_capacity", ...],
        ...     data_types={"passenger_capacity": "int64", ...}
        ... )
    """
    # Check for missing columns
    missing_columns = set(expected_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Check data types
    for col, expected_dtype in data_types.items():
        if col in data.columns:
            actual_dtype = str(data[col].dtype)
            if actual_dtype != expected_dtype:
                logger.warning(
                    f"Column '{col}' has dtype {actual_dtype}, expected {expected_dtype}"
                )

    # Check for all-NaN columns
    for col in expected_columns:
        if data[col].isna().all():
            raise ValueError(f"Column '{col}' is entirely NaN")

    logger.info(f"Schema validation passed: {len(data)} rows, {len(data.columns)} columns")


def configure_logging(
    logger_name: str = "aircraft_range_pipeline",
    level: str = "INFO",
) -> logging.Logger:
    """
    Configure structured logging for data science pipeline.

    Args:
        logger_name: Name for logger instance
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance

    Example:
        >>> log = configure_logging("aircraft_range_pipeline", "DEBUG")
        >>> log.info(f"Training model with batch_size=32, learning_rate=0.001")
    """
    log = logging.getLogger(logger_name)
    log.setLevel(level)

    # Console handler with structured format
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)

    return log


# Initialize pipeline logger
pipeline_logger = configure_logging("aircraft_range_pipeline", "INFO")
