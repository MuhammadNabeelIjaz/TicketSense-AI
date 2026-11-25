"""
Authoritative Live Application Screenshot Capture Script for TicketSense-AI
Interacts with the live Streamlit server on http://localhost:8501 using Playwright and Google Chrome.
Captures genuine, high-resolution screenshots for all operational application workflows.
"""

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "screenshots" / "Application_Overview"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CHROME_PATH = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://localhost:8501"

SAMPLE_OUTAGE_TICKET = (
    "Emergency: Primary database cluster db-prod-01 crashed unexpectedly. "
    "All customer checkout transactions are failing nationwide across web and mobile. "
    "Immediate incident response escalation required."
)

SAMPLE_CSV_PATH = PROJECT_ROOT / "tests" / "test_data" / "test_batch_clean.csv"


def capture_all():
    print("[INFO] Starting Playwright browser automation with Chrome...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--window-size=1920,1080"],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1.25,
        )
        page = context.new_page()

        # -------------------------------------------------------------
        # 1. Main Application Interface
        # -------------------------------------------------------------
        print("[INFO] Step 1: Navigating to Main Interface...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("h1", timeout=15000)
        time.sleep(3)

        # Ensure System Architecture expander is open
        expanders = page.query_selector_all("details")
        for exp in expanders:
            if not exp.get_attribute("open"):
                summary = exp.query_selector("summary")
                if summary:
                    summary.click()
                    time.sleep(0.5)

        screen1_path = OUTPUT_DIR / "01_main_application_interface.png"
        page.screenshot(path=str(screen1_path), full_page=True)
        print(f"[SUCCESS] Saved Screen 1: {screen1_path.name} ({screen1_path.stat().st_size} bytes)")

        # -------------------------------------------------------------
        # 2. Single Ticket Prediction & Complete Breakdown Table
        # -------------------------------------------------------------
        print("[INFO] Step 2: Running Single Ticket Prediction...")
        page.locator("[role='tab']").nth(0).click()
        time.sleep(1)

        textarea = page.wait_for_selector("textarea", timeout=10000)
        textarea.fill("")
        textarea.fill(SAMPLE_OUTAGE_TICKET)
        time.sleep(1)

        # Click Classify Ticket button
        page.locator("button:has-text('Classify Ticket')").click()
        print("[INFO] Clicked Classify Ticket, waiting for model inference...")
        time.sleep(4)

        # Ensure the table is rendered
        page.wait_for_selector("div[data-testid='stDataFrame']", timeout=15000)
        time.sleep(2)

        # Scroll to show metrics and breakdown table
        page.evaluate("window.scrollTo(0, 200)")
        time.sleep(1)

        screen2_path = OUTPUT_DIR / "02_single_ticket_prediction_and_table.png"
        page.screenshot(path=str(screen2_path), full_page=True)
        print(f"[SUCCESS] Saved Screen 2: {screen2_path.name} ({screen2_path.stat().st_size} bytes)")

        # -------------------------------------------------------------
        # 3. Human Review & Correction Flow
        # -------------------------------------------------------------
        print("[INFO] Step 3: Capturing Human Review & Correction Panel...")
        correction_form = page.wait_for_selector("div[data-testid='stForm']", timeout=10000)
        time.sleep(1)

        screen3_path = OUTPUT_DIR / "03_human_review_and_correction.png"
        correction_form.screenshot(path=str(screen3_path))
        print(f"[SUCCESS] Saved Screen 3: {screen3_path.name} ({screen3_path.stat().st_size} bytes)")

        # -------------------------------------------------------------
        # 4. Batch CSV Processing Tab
        # -------------------------------------------------------------
        print("[INFO] Step 4: Interacting with Batch Processing Tab...")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        page.locator("[role='tab']").nth(1).click()
        time.sleep(2)

        # Upload batch CSV (input is hidden, so locate and set files directly)
        file_input = page.locator("input[type='file']")
        file_input.set_input_files(str(SAMPLE_CSV_PATH))
        print(f"[INFO] Uploaded {SAMPLE_CSV_PATH.name}, waiting for Process button...")
        time.sleep(3)

        # Click Process Batch Predictions button
        batch_btn = page.wait_for_selector("button:has-text('Process Batch Predictions')", timeout=15000)
        batch_btn.click()
        print("[INFO] Clicked Process Batch Predictions, computing predictions...")
        
        # Wait for unique completed batch download button to appear
        page.wait_for_selector("button:has-text('Download Enriched Predictions (CSV)')", timeout=30000)
        time.sleep(3)

        screen4_path = OUTPUT_DIR / "04_batch_csv_processing.png"
        page.screenshot(path=str(screen4_path), full_page=True)
        print(f"[SUCCESS] Saved Screen 4: {screen4_path.name} ({screen4_path.stat().st_size} bytes)")

        # -------------------------------------------------------------
        # 5. Performance & Governance Dashboard
        # -------------------------------------------------------------
        print("[INFO] Step 5: Interacting with Performance & Governance Dashboard...")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        page.locator("[role='tab']").nth(2).click()
        
        # Wait for governance header and charts
        page.wait_for_selector("text=Production Models Evaluation Scorecard", timeout=20000)
        page.wait_for_selector("text=Governance & Priority Model Policy", timeout=20000)
        time.sleep(4)  # Wait for charts to finish rendering

        screen5_path = OUTPUT_DIR / "05_dashboard_and_visualizations.png"
        page.screenshot(path=str(screen5_path), full_page=True)
        print(f"[SUCCESS] Saved Screen 5: {screen5_path.name} ({screen5_path.stat().st_size} bytes)")

        browser.close()
        print("[INFO] All live screenshots captured and verified successfully!")


if __name__ == "__main__":
    capture_all()
