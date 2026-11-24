"""Dataset loading utilities for training, validation, and testing."""

from typing import Tuple, Dict
from pathlib import Path
import pandas as pd
from ticketsense.config import (
    TRAIN_VAL_DATASET_PATH,
    TEST_DATASET_PATH,
    SYNTHETIC_DATA_PATH,
    APPROVED_CATEGORIES,
)


def load_dataset_splits(
    file_path: Path = TRAIN_VAL_DATASET_PATH,
    include_synthetic: bool = True,
    synthetic_path: Path = SYNTHETIC_DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and validation splits from the processed dataset.

    Args:
        file_path: Path to the primary processed dataset.
        include_synthetic: Whether to include validated synthetic expansion data in train split.
        synthetic_path: Path to the synthetic expansion CSV.

    Returns:
        Tuple of (train_df: pd.DataFrame, val_df: pd.DataFrame).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {file_path}")

    df = pd.read_csv(file_path, encoding="utf-8")

    # Assert required columns exist
    required_cols = ["ticket_id", "ticket_text", "category", "split"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' missing from {file_path}")

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    val_df = df[df["split"] == "validation"].copy().reset_index(drop=True)

    # Incorporate validated synthetic expansion records into training split if available
    if include_synthetic and synthetic_path.exists():
        syn_df = pd.read_csv(synthetic_path, encoding="utf-8")
        syn_train = syn_df[syn_df["split"] == "train"].copy().reset_index(drop=True)
        train_df = pd.concat([train_df, syn_train], ignore_index=True)

    # Sanity checks
    if len(train_df) == 0:
        raise ValueError("Train split is empty.")
    if len(val_df) == 0:
        raise ValueError("Validation split is empty.")

    return train_df, val_df


def load_test_dataset(
    file_path: Path = TEST_DATASET_PATH
) -> pd.DataFrame:
    """Load the test split for final evaluation.

    Note: As per project guidelines, this dataset is 100% synthetic and
    must NOT be used during iterative model hyperparameter tuning.

    Returns:
        pd.DataFrame containing test records.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {file_path}")

    test_df = pd.read_csv(file_path, encoding="utf-8")
    return test_df.copy().reset_index(drop=True)
