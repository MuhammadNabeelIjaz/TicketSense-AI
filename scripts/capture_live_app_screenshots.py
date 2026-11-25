"""
Automated Live Application Screenshot Capture Script for TicketSense-AI
Uses Playwright with local Chrome to interact with the live running Streamlit app (http://localhost:8501)
and capture high-resolution production UI screenshots into screenshots/Application_Overview/.
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
    print("[INFO] Starting Playwright browser automation...")
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_PATH,
            headless=True,
            args=["--no-sandbox", "--disable-gpu", "--window-size=1920,1080"],
        )
        context = browser.new_context(
            viewport={"width": 1920, "height": 1080},
            device_scale_factor=1.5,
        )
        page = context.new_page()

        # -------------------------------------------------------------
        # 1. Main Application Interface
        # -------------------------------------------------------------
        print("[INFO] Navigating to Streamlit app for Screen 1 (Main Interface)...")
        page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_selector("h1", timeout=15000)
        time.sleep(3)  # Let Streamlit animations stabilize

        # Expand system architecture expander if present
        expander = page.query_selector("div[data-testid='stExpander']")
        if expander:
            try:
                expander.click()
                time.sleep(1)
            except Exception as e:
                print(f"[WARN] Expander click: {e}")

        screen1_path = OUTPUT_DIR / "01_main_application_interface.png"
        page.screenshot(path=str(screen1_path), full_page=True)
        print(f"[SUCCESS] Saved Screen 1: {screen1_path}")

        # -------------------------------------------------------------
        # 2. Single Ticket Prediction & Complete Table Breakdown
        # -------------------------------------------------------------
        print("[INFO] Interacting with Single Ticket Triage for Screen 2...")
        # Ensure Single Ticket tab is selected
        tabs = page.query_selector_all("button[data-testid='stTab']")
        if tabs:
            tabs[0].click()
            time.sleep(1)

        # Clear text area and enter sample outage ticket
        textarea = page.wait_for_selector("textarea", timeout=10000)
        textarea.fill("")
        textarea.fill(SAMPLE_OUTAGE_TICKET)
        time.sleep(1)

        # Click "Classify Ticket" button
        classify_button = None
        buttons = page.query_selector_all("button")
        for b in buttons:
            if "Classify Ticket" in (b.inner_text() or ""):
                classify_button = b
                break

        if classify_button:
            classify_button.click()
            print("[INFO] Clicked Classify Ticket, waiting for inference results...")
            time.sleep(4)  # Wait for ML inference and DOM update

        # Wait for dataframe or result containers
        page.wait_for_selector("div[data-testid='stDataFrame']", timeout=10000)
        time.sleep(1)

        screen2_path = OUTPUT_DIR / "02_single_ticket_prediction_and_table.png"
        page.screenshot(path=str(screen2_path), full_page=True)
        print(f"[SUCCESS] Saved Screen 2: {screen2_path}")

        # -------------------------------------------------------------
        # 3. Human Review & Correction Flow
        # -------------------------------------------------------------
        print("[INFO] Capturing Screen 3 (Human Review & Correction)...")
        # Scroll to human-in-the-loop correction section
        correction_header = page.query_selector("text=Human-in-the-Loop Feedback & Correction")
        if correction_header:
            correction_header.scroll_into_view_if_needed()
            time.sleep(1)

        screen3_path = OUTPUT_DIR / "03_human_review_and_correction.png"
        # Capture current viewport or full page
        page.screenshot(path=str(screen3_path), full_page=True)
        print(f"[SUCCESS] Saved Screen 3: {screen3_path}")

        # -------------------------------------------------------------
        # 4. Batch CSV Processing Tab
        # -------------------------------------------------------------
        print("[INFO] Navigating to Batch Processing for Screen 4...")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        tabs = page.query_selector_all("button[data-testid='stTab']")
        if len(tabs) >= 2:
            tabs[1].click()
            time.sleep(2)

            # Upload CSV file
            file_input = page.query_selector("input[type='file']")
            if file_input and SAMPLE_CSV_PATH.exists():
                file_input.set_input_files(str(SAMPLE_CSV_PATH))
                print(f"[INFO] Uploaded {SAMPLE_CSV_PATH.name}, waiting for processing...")
                time.sleep(3)

                # Click "Process Batch Predictions" if button exists
                batch_btns = page.query_selector_all("button")
                for b in batch_btns:
                    if "Process Batch Predictions" in (b.inner_text() or ""):
                        b.click()
                        print("[INFO] Clicked Process Batch Predictions...")
                        time.sleep(4)
                        break

            time.sleep(2)

        screen4_path = OUTPUT_DIR / "04_batch_csv_processing.png"
        page.screenshot(path=str(screen4_path), full_page=True)
        print(f"[SUCCESS] Saved Screen 4: {screen4_path}")

        # -------------------------------------------------------------
        # 5. Performance & Governance Dashboard
        # -------------------------------------------------------------
        print("[INFO] Navigating to Performance & Governance for Screen 5...")
        page.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        tabs = page.query_selector_all("button[data-testid='stTab']")
        if len(tabs) >= 3:
            tabs[2].click()
            time.sleep(3)

        screen5_path = OUTPUT_DIR / "05_dashboard_and_visualizations.png"
        page.screenshot(path=str(screen5_path), full_page=True)
        print(f"[SUCCESS] Saved Screen 5: {screen5_path}")

        browser.close()
        print("[INFO] Browser automation finished successfully!")


if __name__ == "__main__":
    capture_all()
