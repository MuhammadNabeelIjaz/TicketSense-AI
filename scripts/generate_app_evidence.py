import shutil
from pathlib import Path
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = PROJECT_ROOT / "docs" / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

VISUALIZATIONS_DIR = PROJECT_ROOT / "reports" / "visualizations"

def generate_evidence():
    print("Generating application and reporting screenshots/visual evidence...")
    
    # 1. Capture live app screenshot via Chrome headless
    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    target_img = SCREENSHOTS_DIR / "01_main_application.png"
    cmd = [
        chrome_path,
        "--headless=new",
        f"--screenshot={str(target_img)}",
        "--window-size=1280,960",
        "--virtual-time-budget=8000",
        "http://localhost:8501"
    ]
    try:
        subprocess.run(cmd, check=True, timeout=20, capture_output=True)
        print(f"Captured: {target_img.name} ({target_img.stat().st_size} bytes)")
    except Exception as e:
        print(f"Failed to capture live Chrome screenshot: {e}")

    # 2. Copy all report visualizations into docs/screenshots for easy markdown referencing
    mapping = {
        "dataset_class_distribution.png": "02_dataset_class_distribution.png",
        "category_performance.png": "03_category_performance.png",
        "test_confusion_matrix.png": "04_test_confusion_matrix.png",
        "confidence_distribution.png": "05_confidence_distribution.png",
        "automatic_vs_manual_review.png": "06_automatic_vs_manual_review.png",
        "model_experiment_comparison.png": "07_model_experiment_comparison.png",
    }
    
    for src_name, dst_name in mapping.items():
        src = VISUALIZATIONS_DIR / src_name
        dst = SCREENSHOTS_DIR / dst_name
        if src.exists():
            shutil.copy2(src, dst)
            print(f"Copied visualization: {src_name} -> docs/screenshots/{dst_name}")
        else:
            print(f"Source visualization missing: {src}")

    print("Evidence generation complete.")

if __name__ == "__main__":
    generate_evidence()
