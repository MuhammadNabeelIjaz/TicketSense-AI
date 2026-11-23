"""Machine learning modeling and evaluation package for TicketSense-AI."""

from ticketsense.models.persistence import save_model_bundle, load_model_bundle
from ticketsense.models.evaluator import evaluate_classification, analyze_errors
from ticketsense.models.category_pipeline import build_category_pipeline, train_category_model

__all__ = [
    "save_model_bundle",
    "load_model_bundle",
    "evaluate_classification",
    "analyze_errors",
    "build_category_pipeline",
    "train_category_model",
]
