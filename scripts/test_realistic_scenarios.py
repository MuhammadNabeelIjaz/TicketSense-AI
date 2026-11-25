import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ticketsense.inference.engine import InferenceEngine

def run_tests():
    engine = InferenceEngine()
    assert engine.is_ready(), "InferenceEngine failed to load model bundles"

    test_scenarios = [
        ("Billing dispute", "I was double charged on my invoice #98234 last week and need an immediate refund."),
        ("Technical issue", "The dashboard is throwing 500 internal server error when loading analytics."),
        ("Account access", "I cannot log into my account. Password reset email is never arriving."),
        ("Delivery issue", "Tracking says my package was delivered yesterday but nothing arrived at my address."),
        ("General inquiry", "What are your business operating hours and pricing tiers for small teams?"),
        ("High-impact outage", "Emergency: our primary database cluster crashed and customer checkout is completely down nationwide. Immediate escalation needed."),
        ("Ambiguous ticket", "My account was billed twice but I also cannot log in to download my invoice."),
        ("Misspelled ticket", "plz halp my accont is loged out and cant acces billing recipts"),
        ("Short text", "help"),
        ("Empty text", ""),
        ("Out-of-category message", "Can you recommend a good Italian restaurant in downtown Chicago?")
    ]

    print("================================================================================")
    print("REALISTIC TEST SCENARIOS VERIFICATION")
    print("================================================================================")

    for name, text in test_scenarios:
        print(f"\n--- Scenario: {name} ---")
        print(f"Input: {repr(text)}")
        result = engine.predict_ticket(text)
        
        if not result["success"]:
            print(f"Status: REJECTED | Error: {result.get('error')}")
            continue

        print(f"Category: {result['predicted_category']} (Confidence: {result['confidence']:.4f}, Review: {result['category_review_required']})")
        if result["category_review_required"]:
            print(f"  Category Review Reason: {result['uncertainty_reason']}")
        
        print(f"Priority: {result['priority']} (Impact: {result['predicted_impact']}, Urgency: {result['predicted_urgency']}, Review: {result['priority_review_required']})")
        if result["priority_review_required"]:
            print(f"  Priority Review Reason: {result['priority_uncertainty_reason']}")
        
        print(f"Overall Review Required: {result['overall_review_required']}")

    print("\n================================================================================")
    print("VERIFICATION COMPLETED")
    print("================================================================================")

if __name__ == "__main__":
    run_tests()
