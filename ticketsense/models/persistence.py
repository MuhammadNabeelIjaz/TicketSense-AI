"""Model persistence and serialization utilities using joblib and JSON metadata."""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
from ticketsense.config import CATEGORY_MODEL_PATH, CATEGORY_METADATA_PATH


def save_model_bundle(
    pipeline: Any,
    metadata: Dict[str, Any],
    model_path: Path = CATEGORY_MODEL_PATH,
    metadata_path: Path = CATEGORY_METADATA_PATH,
) -> None:
    """Serialize model pipeline and metadata to disk.

    Args:
        pipeline: Trained scikit-learn Pipeline or estimator.
        metadata: Dictionary containing training parameters, metrics, timestamp.
        model_path: Path to save the serialized pipeline.
        metadata_path: Path to save the metadata JSON.
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, model_path)

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def load_model_bundle(
    model_path: Path = CATEGORY_MODEL_PATH,
    metadata_path: Path = CATEGORY_METADATA_PATH,
) -> Tuple[Any, Dict[str, Any]]:
    """Load serialized model pipeline and metadata from disk.

    Args:
        model_path: Path to serialized pipeline.
        metadata_path: Path to metadata JSON.

    Returns:
        Tuple of (pipeline, metadata).
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. "
            f"Please run the training script first."
        )

    pipeline = joblib.load(model_path)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    return pipeline, metadata
