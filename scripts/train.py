"""CLI runner to execute model training and generate evaluation artifacts."""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ticketsense.models.category_pipeline import train_category_model


def main():
    print("=" * 60)
    print("TicketSense-AI Category Model Training Runner")
    print("=" * 60)
    print("Training on 'train' split (400 original + 200 synthetic = 600 samples)...")
    print("Evaluating on 'validation' split (100 samples)...")
    print("Note: Test split (100 samples) remains isolated.")

    results = train_category_model(save_artifacts=True)
    meta = results["metadata"]
    metrics = results["metrics"]

    print("\n--- Training Completed Successfully ---")
    print(f"Model Architecture:  {meta['model_name']}")
    print(f"Training Samples:    {meta['train_samples']}")
    print(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1-Score:      {metrics['macro_f1']:.4f}")
    print(f"Weighted F1-Score:   {metrics['weighted_f1']:.4f}")
    print(f"Validation Errors:   {meta['errors_count']} / {meta['val_samples']}")
    print("\nArtifacts saved:")
    print(" - Model:   models/category_model.joblib")
    print(" - Meta:    models/category_model_metadata.json")
    print(" - Report:  reports/CATEGORY_MODEL_EVALUATION.md")
    print(" - Plot:    reports/category_confusion_matrix.png")
    print(" - Errors:  reports/category_validation_errors.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
