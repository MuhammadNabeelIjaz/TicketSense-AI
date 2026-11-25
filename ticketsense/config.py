"""TicketSense-AI Configuration Module.

Central configuration for paths, category taxonomy, priority policy,
uncertainty thresholds, and modeling parameters.
"""

from pathlib import Path
from typing import List, Dict, Any

# Base Directories
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
CORRECTIONS_DIR: Path = DATA_DIR / "corrections"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
VISUALIZATIONS_DIR: Path = REPORTS_DIR / "visualizations"
FIGURES_DIR: Path = VISUALIZATIONS_DIR
TESTS_DIR: Path = PROJECT_ROOT / "tests"

# Dataset Files
TRAIN_VAL_DATASET_PATH: Path = PROCESSED_DATA_DIR / "ticketsense_dataset.csv"
TEST_DATASET_PATH: Path = PROCESSED_DATA_DIR / "ticketsense_test.csv"
SYNTHETIC_DATA_PATH: Path = PROCESSED_DATA_DIR / "ticketsense_synthetic_expansion_v1.csv"
RUNTIME_CORRECTIONS_PATH: Path = CORRECTIONS_DIR / "runtime_corrections.csv"
REVIEW_QUEUE_PATH: Path = PROCESSED_DATA_DIR / "dataset_review_required.csv"

# Model Artifacts
CATEGORY_MODEL_PATH: Path = MODELS_DIR / "category_model.joblib"
CATEGORY_METADATA_PATH: Path = MODELS_DIR / "category_model_metadata.json"

# Approved Taxonomies
APPROVED_CATEGORIES: List[str] = [
    "Account Access",
    "Billing",
    "Delivery",
    "General Inquiry",
    "Technical Issue",
]

APPROVED_PRIORITIES: List[str] = [
    "Low",
    "Medium",
    "High",
]

# Policy Guardrails
# Priority training is strictly PAUSED pending Gate G1 supervisor approval
PRIORITY_TRAINING_PAUSED: bool = True
PRIORITY_POLICY_STATUS_MESSAGE: str = (
    "Priority training paused: labels provisional (6.5% High minority) "
    "pending Gate G1 supervisor sign-off."
)

# Inference & Uncertainty Defaults
DEFAULT_UNCERTAINTY_THRESHOLD: float = 0.65
DEFAULT_UNCERTAINTY_MARGIN: float = 0.15
DEFAULT_PRIORITY_UNCERTAINTY_THRESHOLD: float = 0.60
DEFAULT_PRIORITY_UNCERTAINTY_MARGIN: float = 0.15
DEFAULT_PRIORITY_COMPONENT_THRESHOLD: float = 0.50
DEFAULT_PRIORITY_HIGH_RISK_THRESHOLD: float = 0.15
MIN_TICKET_CHAR_LENGTH: int = 5
MAX_TICKET_CHAR_LENGTH: int = 5000

# Reproducibility
RANDOM_STATE: int = 42
