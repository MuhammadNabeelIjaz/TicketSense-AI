"""Priority classification inference module.

Loads the trained Priority ML model (Two-Stage Rubric Model)
and predicts ticket urgency (Low, Medium, High) along with Impact and Urgency levels.
Provides graceful fallback to the deterministic keyword heuristic if model
artifacts are unavailable. Incorporates independent Priority uncertainty quantification.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import joblib

from ticketsense.config import (
    MODELS_DIR,
    APPROVED_PRIORITIES,
    DEFAULT_PRIORITY_UNCERTAINTY_THRESHOLD,
    DEFAULT_PRIORITY_UNCERTAINTY_MARGIN,
    DEFAULT_PRIORITY_COMPONENT_THRESHOLD,
    DEFAULT_PRIORITY_HIGH_RISK_THRESHOLD,
)
from ticketsense.inference.priority_stub import assess_priority
from ticketsense.inference.uncertainty import compute_priority_uncertainty_score

DEFAULT_PRIORITY_MODEL_PATH: Path = MODELS_DIR / "priority_model.joblib"
DEFAULT_PRIORITY_METADATA_PATH: Path = MODELS_DIR / "priority_model_metadata.json"


class PriorityPredictor:
    """Inference engine for priority classification."""

    def __init__(
        self,
        model_path: Path = DEFAULT_PRIORITY_MODEL_PATH,
        metadata_path: Path = DEFAULT_PRIORITY_METADATA_PATH,
        uncertainty_threshold: float = DEFAULT_PRIORITY_UNCERTAINTY_THRESHOLD,
        uncertainty_margin: float = DEFAULT_PRIORITY_UNCERTAINTY_MARGIN,
        component_threshold: float = DEFAULT_PRIORITY_COMPONENT_THRESHOLD,
        high_risk_threshold: float = DEFAULT_PRIORITY_HIGH_RISK_THRESHOLD,
    ):
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        self.uncertainty_threshold = uncertainty_threshold
        self.uncertainty_margin = uncertainty_margin
        self.component_threshold = component_threshold
        self.high_risk_threshold = high_risk_threshold
        self.pipeline = None
        self.metadata = {}
        self.classes: List[str] = APPROVED_PRIORITIES
        self._load_model()

    def _load_model(self) -> None:
        """Load trained priority model pipeline and metadata from disk."""
        if self.model_path.exists():
            try:
                self.pipeline = joblib.load(self.model_path)
                if hasattr(self.pipeline, "classes_"):
                    self.classes = list(self.pipeline.classes_)
                if self.metadata_path.exists():
                    import json
                    self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception as e:
                self.pipeline = None
                self.metadata = {"load_error": str(e)}

    def is_ready(self) -> bool:
        """Check if priority ML pipeline is loaded and ready."""
        return self.pipeline is not None

    def predict(self, ticket_text: str) -> Dict[str, Any]:
        """Predict priority and class probabilities for input message.

        Args:
            ticket_text: Support ticket message.

        Returns:
            Dict containing predicted priority, probabilities, confidence,
            governance status, heuristic fallback, and independent uncertainty diagnostics.
        """
        if not isinstance(ticket_text, str):
            ticket_text = str(ticket_text) if ticket_text is not None else ""

        text = ticket_text.strip()
        heuristic_res = assess_priority(text)

        if not self.is_ready() or not text:
            # Fallback to heuristic if model is not loaded or text is empty
            p_val = heuristic_res["heuristic_priority"]
            p_diag = compute_priority_uncertainty_score(
                priority=p_val,
                priority_probabilities={p_val: 1.0},
                predicted_impact=None,
                impact_confidence=None,
                predicted_urgency=None,
                urgency_confidence=None,
                ticket_text=text,
                threshold=self.uncertainty_threshold,
                margin=self.uncertainty_margin,
                component_threshold=self.component_threshold,
                high_risk_threshold=self.high_risk_threshold,
            )
            # Fallback or empty input always requires human review
            p_diag["priority_review_required"] = True
            p_diag["reason"] = "Model not loaded or empty input; degraded to heuristic; requires review."

            return {
                "priority": p_val,
                "priority_confidence": None,
                "priority_probabilities": {},
                "priority_margin": None,
                "priority_review_required": True,
                "priority_uncertainty_reason": p_diag["reason"],
                "priority_review_details": p_diag,
                "predicted_impact": None,
                "impact_confidence": None,
                "impact_probabilities": {},
                "predicted_urgency": None,
                "urgency_confidence": None,
                "urgency_probabilities": {},
                "priority_status": "FALLBACK_HEURISTIC",
                "is_model_trained": False,
                "policy_message": heuristic_res["policy_message"],
                "heuristic_fallback": heuristic_res["heuristic_priority"],
                "heuristic_notes": heuristic_res["heuristic_notes"],
            }

        # Predict with trained Two-Stage ML pipeline
        try:
            if hasattr(self.pipeline, "predict_detailed"):
                detailed = self.pipeline.predict_detailed([text])[0]
                p_val = detailed["priority"]
                p_probs = detailed["priority_probabilities"]
                imp_val = detailed["predicted_impact"]
                imp_conf = detailed["impact_confidence"]
                urg_val = detailed["predicted_urgency"]
                urg_conf = detailed["urgency_confidence"]

                p_diag = compute_priority_uncertainty_score(
                    priority=p_val,
                    priority_probabilities=p_probs,
                    predicted_impact=imp_val,
                    impact_confidence=imp_conf,
                    predicted_urgency=urg_val,
                    urgency_confidence=urg_conf,
                    ticket_text=text,
                    threshold=self.uncertainty_threshold,
                    margin=self.uncertainty_margin,
                    component_threshold=self.component_threshold,
                    high_risk_threshold=self.high_risk_threshold,
                )

                return {
                    "priority": p_val,
                    "priority_confidence": detailed["priority_confidence"],
                    "priority_probabilities": p_probs,
                    "priority_margin": p_diag["priority_margin"],
                    "priority_review_required": p_diag["priority_review_required"],
                    "priority_uncertainty_reason": p_diag["reason"],
                    "priority_review_details": p_diag,
                    "predicted_impact": imp_val,
                    "impact_confidence": imp_conf,
                    "impact_probabilities": detailed["impact_probabilities"],
                    "predicted_urgency": urg_val,
                    "urgency_confidence": urg_conf,
                    "urgency_probabilities": detailed["urgency_probabilities"],
                    "priority_status": "ACTIVE_ML_MODEL",
                    "is_model_trained": True,
                    "policy_message": (
                        "Two-Stage ML Priority Model (Impact x Urgency Rubric). "
                        "Stage 1: Impact Classifier; Stage 2: Urgency Classifier; Stage 3: Authoritative Matrix Mapping."
                    ),
                    "heuristic_fallback": heuristic_res["heuristic_priority"],
                    "heuristic_notes": heuristic_res["heuristic_notes"],
                }
            else:
                probas = self.pipeline.predict_proba([text])[0]
                class_probas = {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes, probas)}
                top_priority = str(self.pipeline.predict([text])[0])
                top_confidence = round(float(max(probas)), 4)

                p_diag = compute_priority_uncertainty_score(
                    priority=top_priority,
                    priority_probabilities=class_probas,
                    predicted_impact=None,
                    impact_confidence=None,
                    predicted_urgency=None,
                    urgency_confidence=None,
                    ticket_text=text,
                    threshold=self.uncertainty_threshold,
                    margin=self.uncertainty_margin,
                    component_threshold=self.component_threshold,
                    high_risk_threshold=self.high_risk_threshold,
                )

                return {
                    "priority": top_priority,
                    "priority_confidence": top_confidence,
                    "priority_probabilities": class_probas,
                    "priority_margin": p_diag["priority_margin"],
                    "priority_review_required": p_diag["priority_review_required"],
                    "priority_uncertainty_reason": p_diag["reason"],
                    "priority_review_details": p_diag,
                    "predicted_impact": None,
                    "impact_confidence": None,
                    "impact_probabilities": {},
                    "predicted_urgency": None,
                    "urgency_confidence": None,
                    "urgency_probabilities": {},
                    "priority_status": "ACTIVE_ML_MODEL",
                    "is_model_trained": True,
                    "policy_message": (
                        "Active ML Priority Model. "
                        "Note: Labels are rubric-derived (Impact x Urgency matrix)."
                    ),
                    "heuristic_fallback": heuristic_res["heuristic_priority"],
                    "heuristic_notes": heuristic_res["heuristic_notes"],
                }
        except Exception as e:
            # Safe degradation to heuristic
            p_val = heuristic_res["heuristic_priority"]
            return {
                "priority": p_val,
                "priority_confidence": None,
                "priority_probabilities": {},
                "priority_margin": None,
                "priority_review_required": True,
                "priority_uncertainty_reason": f"Inference degraded to heuristic due to error: {e}",
                "priority_review_details": {"priority_review_required": True, "reason": str(e)},
                "predicted_impact": None,
                "impact_confidence": None,
                "impact_probabilities": {},
                "predicted_urgency": None,
                "urgency_confidence": None,
                "urgency_probabilities": {},
                "priority_status": "FALLBACK_HEURISTIC",
                "is_model_trained": False,
                "policy_message": f"Inference degraded to heuristic due to error: {e}",
                "heuristic_fallback": heuristic_res["heuristic_priority"],
                "heuristic_notes": heuristic_res["heuristic_notes"],
            }
