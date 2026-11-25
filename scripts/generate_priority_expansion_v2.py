"""Generate, validate, and document a synthetic Priority training dataset expansion (v2).

Expands coverage of underrepresented High (I3|U3, I3|U2, I2|U3) and Medium (I3|U1, I2|U2, I1|U3) classes
with diverse phrasing, implicit urgency, realistic business scenarios, and explicit rubric justifications.
Validates zero data leakage against validation and locked test sets.
"""

import sys
from pathlib import Path
import pandas as pd
import hashlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ticketsense.models.priority_pipeline import AUTHORITATIVE_RUBRIC

# 50 Carefully authored synthetic records across realistic scenarios
SYNTHETIC_RECORDS = [
    # ---------------- High Priority (I3|U3, I3|U2, I2|U3) ----------------
    {
        "ticket_id": "TS-PRI-EXP2-0001",
        "ticket_text": "Our payment processing webhook is rejecting all incoming card authorizations. Zero transactions have processed for the past 45 minutes across all store locations.",
        "category": "Technical Issue", "impact": "I3", "urgency": "U3",
        "scenario_type": "revenue_halting_infrastructure",
        "impact_reason": "Complete company-wide payment processing failure affecting all retail locations.",
        "urgency_reason": "Active revenue loss occurring in real time requires instant resolution."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0002",
        "ticket_text": "I see three unauthorized wire transfers dispatched from my corporate treasury portal that nobody on our finance team approved. Freeze the portal immediately.",
        "category": "Billing", "impact": "I3", "urgency": "U3",
        "scenario_type": "active_fraudulent_transactions",
        "impact_reason": "High-value unauthorized fund movement from corporate treasury.",
        "urgency_reason": "Active fraud in progress requiring immediate portal freeze."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0003",
        "ticket_text": "Temperature-controlled insulin shipment tracking shows the package was left on an unshaded loading dock in 95 degree heat. Courier must move it into refrigeration right now.",
        "category": "Delivery", "impact": "I2", "urgency": "U3",
        "scenario_type": "time_critical_medical_delivery",
        "impact_reason": "High single-customer risk of spoiled vital medical medication.",
        "urgency_reason": "Thermal degradation happens within hours, demanding immediate intervention."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0004",
        "ticket_text": "Every user trying to access the customer support knowledge base is receiving an SSL certificate expired error. Public trust and incoming support are severely impacted.",
        "category": "Technical Issue", "impact": "I3", "urgency": "U2",
        "scenario_type": "public_trust_ssl_failure",
        "impact_reason": "Company-wide public-facing security warning visible to all visitors.",
        "urgency_reason": "Must be renewed and deployed today before end-of-business."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0005",
        "ticket_text": "I have an emergency board meeting in thirty minutes and my credentials were unexpectedly revoked during an automated security patch. Please restore access.",
        "category": "Account Access", "impact": "I2", "urgency": "U3",
        "scenario_type": "executive_time_critical_lockout",
        "impact_reason": "Disrupts critical executive governance and fiduciary responsibility.",
        "urgency_reason": "Imminent deadline in 30 minutes makes resolution time-critical."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0006",
        "ticket_text": "Our enterprise contract renewal pricing engine generated quotes with a 90% discount across forty accounts. We need the quotation generator halted before sales sends them.",
        "category": "Billing", "impact": "I3", "urgency": "U3",
        "scenario_type": "contract_pricing_anomaly",
        "impact_reason": "Severe contractual and revenue exposure across multiple accounts.",
        "urgency_reason": "Quotes actively being sent out by sales team right now."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0007",
        "ticket_text": "The delivery address for our trade show booth materials was routed to the wrong convention hall. The exhibition starts tomorrow morning at 8 AM.",
        "category": "Delivery", "impact": "I2", "urgency": "U3",
        "scenario_type": "event_logistics_deadline",
        "impact_reason": "Commercial loss and public absence from high-value exhibition.",
        "urgency_reason": "Hard physical deadline tomorrow morning at 8 AM."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0008",
        "ticket_text": "A database migration script dropped the foreign key constraints on our user table. Data corruption is accumulating on new writes. Halt the application server.",
        "category": "Technical Issue", "impact": "I3", "urgency": "U3",
        "scenario_type": "data_corruption_production",
        "impact_reason": "System-wide data corruption multiplying on every incoming write.",
        "urgency_reason": "Active data degradation requires immediate process termination."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0009",
        "ticket_text": "Our automated tax filing integration submitted Q3 filings with duplicated payroll expenses. We need manual assistance to submit an amended filing before the statutory deadline today.",
        "category": "Billing", "impact": "I3", "urgency": "U2",
        "scenario_type": "regulatory_compliance_deadline",
        "impact_reason": "Statutory legal and financial penalties from state authorities.",
        "urgency_reason": "Statutory deadline closes at midnight tonight."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0010",
        "ticket_text": "An imposter created a duplicate profile using my photo and is messaging my enterprise contacts requesting password resets. Shut down this fraudulent profile.",
        "category": "Account Access", "impact": "I2", "urgency": "U3",
        "scenario_type": "active_social_engineering",
        "impact_reason": "Impersonation targeting corporate contacts.",
        "urgency_reason": "Active social engineering attack occurring right now."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0011",
        "ticket_text": "Our warehouse fulfillment barcode scanners have lost network synchronization. Hundreds of packages cannot be verified against the outgoing manifest.",
        "category": "Delivery", "impact": "I3", "urgency": "U2",
        "scenario_type": "depot_throughput_bottleneck",
        "impact_reason": "Whole warehouse dispatch workflow stalled.",
        "urgency_reason": "Outgoing trucks scheduled to depart this afternoon."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0012",
        "ticket_text": "Production Kubernetes cluster has twelve pods in CrashLoopBackOff following the 2.4 release. Core analytics service is unavailable for all web clients.",
        "category": "Technical Issue", "impact": "I3", "urgency": "U3",
        "scenario_type": "container_cluster_crash",
        "impact_reason": "Core analytical capability completely offline for entire client base.",
        "urgency_reason": "Production service degraded right now."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0013",
        "ticket_text": "Someone drained my account balance through an unknown API token created at 3 AM. I did not generate this token. Invalidate it and secure my funds.",
        "category": "Account Access", "impact": "I3", "urgency": "U3",
        "scenario_type": "api_credential_compromise",
        "impact_reason": "Complete financial loss and API credential theft.",
        "urgency_reason": "Active breach with API key in unauthorized hands."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0014",
        "ticket_text": "Invoice PDF generator is outputting cleartext bank account numbers of other customers on my invoices. Massive data leak happening on billing downloads.",
        "category": "Billing", "impact": "I3", "urgency": "U3",
        "scenario_type": "pii_financial_data_leak",
        "impact_reason": "Severe regulatory breach involving third-party financial records.",
        "urgency_reason": "Data actively leaking with every invoice downloaded."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0015",
        "ticket_text": "Perishable seafood shipment worth $8,000 is stuck at the central freight hub due to an incorrect carrier bill of lading number. Ice packs will melt by tonight.",
        "category": "Delivery", "impact": "I2", "urgency": "U3",
        "scenario_type": "spoilage_risk_logistics",
        "impact_reason": "High value physical inventory at total loss risk.",
        "urgency_reason": "Physical spoilage occurs if not cleared today."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0016",
        "ticket_text": "Single Sign-On is failing specifically for our North America sales division during annual contract closing week. Over 200 reps cannot access their CRM pipelines.",
        "category": "Account Access", "impact": "I3", "urgency": "U2",
        "scenario_type": "departmental_sso_failure",
        "impact_reason": "Major business disruption affecting an entire corporate regional division.",
        "urgency_reason": "Happening during quarterly contract closing week."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0017",
        "ticket_text": "Our customer-facing web application is experiencing memory leaks causing server latency to exceed twenty seconds for all users. System is nearing total crash.",
        "category": "Technical Issue", "impact": "I3", "urgency": "U2",
        "scenario_type": "cascading_server_instability",
        "impact_reason": "Application-wide performance degradation affecting all connected users.",
        "urgency_reason": "Cascading failure will cause complete outage if not patched today."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0018",
        "ticket_text": "Our monthly recurring revenue reconciliation script failed to debit twelve enterprise clients. Net thirty billing contracts will expire if not processed today.",
        "category": "Billing", "impact": "I3", "urgency": "U2",
        "scenario_type": "enterprise_reconciliation_omission",
        "impact_reason": "High enterprise revenue at risk of contract lapse.",
        "urgency_reason": "Net-30 expiration occurs at end of business today."
    },

    # ---------------- Medium Priority (I3|U1, I2|U2, I1|U3) ----------------
    {
        "ticket_id": "TS-PRI-EXP2-0019",
        "ticket_text": "The monthly financial export report is missing the regional currency conversion footnotes. This needs updating before our quarterly review next month.",
        "category": "Technical Issue", "impact": "I3", "urgency": "U1",
        "scenario_type": "broad_impact_low_urgency_report",
        "impact_reason": "Affects all corporate quarterly financial statements.",
        "urgency_reason": "Review is scheduled for next month; ample lead time."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0020",
        "ticket_text": "I would like to change the delivery address for my custom furniture order before the dispatch window opens next Wednesday.",
        "category": "Delivery", "impact": "I2", "urgency": "U2",
        "scenario_type": "standard_address_modification",
        "impact_reason": "Single moderate-value consumer order.",
        "urgency_reason": "Dispatch window opens next Wednesday."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0021",
        "ticket_text": "Can someone look into why my profile avatar is stretched horizontally on mobile Safari? It looks distorted but the app works fine.",
        "category": "Technical Issue", "impact": "I1", "urgency": "U3",
        "scenario_type": "cosmetic_urgency_request",
        "impact_reason": "Minor personal cosmetic display issue with no functional disruption.",
        "urgency_reason": "Customer requests immediate review before an interview."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0022",
        "ticket_text": "I was billed $45 instead of $35 for my pro subscription renewal. Please credit the $10 difference back to my card.",
        "category": "Billing", "impact": "I2", "urgency": "U2",
        "scenario_type": "routine_overcharge_dispute",
        "impact_reason": "Routine minor billing discrepancy on individual subscription.",
        "urgency_reason": "Standard billing support SLA is adequate."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0023",
        "ticket_text": "The global system audit log shows that IP geo-location tags are defaulting to US-East for all European staff members. Please patch this in the next release.",
        "category": "Technical Issue", "impact": "I3", "urgency": "U1",
        "scenario_type": "telemetry_metadata_inaccuracy",
        "impact_reason": "Affects enterprise audit records across all European personnel.",
        "urgency_reason": "Non-blocking background telemetry issue planned for next release."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0024",
        "ticket_text": "My package tracking number shows in transit for five days without a carrier scan. Can you open an inquiry with FedEx?",
        "category": "Delivery", "impact": "I2", "urgency": "U2",
        "scenario_type": "stalled_package_tracking",
        "impact_reason": "Single customer delivery tracking delay.",
        "urgency_reason": "Standard tracking trace turnaround."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0025",
        "ticket_text": "I need help exporting my personal purchase receipts from 2024 as a consolidated CSV before I file my individual taxes next week.",
        "category": "Billing", "impact": "I2", "urgency": "U2",
        "scenario_type": "annual_receipt_export",
        "impact_reason": "Personal financial records export for single user.",
        "urgency_reason": "Tax filing deadline is next week."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0026",
        "ticket_text": "We need to schedule a migration of 200 archived mailboxes from our legacy tenant to the new domain during next weekend's maintenance window.",
        "category": "Account Access", "impact": "I3", "urgency": "U1",
        "scenario_type": "scheduled_tenant_migration",
        "impact_reason": "High volume of user accounts involved in tenant transfer.",
        "urgency_reason": "Planned for next weekend's scheduled maintenance."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0027",
        "ticket_text": "The dark mode toggle resets to light mode whenever I refresh my browser. Minor annoyance but would appreciate a fix.",
        "category": "Technical Issue", "impact": "I1", "urgency": "U3",
        "scenario_type": "cosmetic_theme_annoyance",
        "impact_reason": "Pure cosmetic preference with zero functional impact.",
        "urgency_reason": "Customer expresses annoyance and asks for prompt attention."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0028",
        "ticket_text": "Our courier delivered the package to my front porch while it was raining and the cardboard box got soggy. The contents inside were undamaged.",
        "category": "Delivery", "impact": "I2", "urgency": "U2",
        "scenario_type": "minor_packaging_damage",
        "impact_reason": "Moderate packaging issue, items inside intact.",
        "urgency_reason": "Feedback regarding courier delivery standards."
    },

    # ---------------- Low Priority (I2|U1, I1|U2, I1|U1) ----------------
    {
        "ticket_id": "TS-PRI-EXP2-0029",
        "ticket_text": "Could you provide information on whether your annual subscription offers any educational discounts for university students?",
        "category": "General Inquiry", "impact": "I1", "urgency": "U1",
        "scenario_type": "general_pricing_inquiry",
        "impact_reason": "Informational request with no operational disruption.",
        "urgency_reason": "No timeline or urgency attached."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0030",
        "ticket_text": "I noticed an old business address listed on our company profile settings. We moved two years ago. Please update it when convenient.",
        "category": "Account Access", "impact": "I2", "urgency": "U1",
        "scenario_type": "routine_profile_address_change",
        "impact_reason": "Corporate account profile data accuracy.",
        "urgency_reason": "Customer specifies 'when convenient'; zero urgency."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0031",
        "ticket_text": "What are your standard business operating hours and support response times during the upcoming Thanksgiving holiday weekend?",
        "category": "General Inquiry", "impact": "I1", "urgency": "U1",
        "scenario_type": "holiday_schedule_inquiry",
        "impact_reason": "General schedule inquiry.",
        "urgency_reason": "Advance informational query for future holiday."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0032",
        "ticket_text": "How do I update the notification email preferences for secondary account team members?",
        "category": "Account Access", "impact": "I1", "urgency": "U1",
        "scenario_type": "how_to_documentation_request",
        "impact_reason": "Routine feature guidance inquiry.",
        "urgency_reason": "Self-service documentation query."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0033",
        "ticket_text": "I am thinking about upgrading our current plan next quarter. Can you send a PDF feature comparison between Growth and Enterprise?",
        "category": "General Inquiry", "impact": "I1", "urgency": "U1",
        "scenario_type": "sales_brochure_request",
        "impact_reason": "Prospective upgrade inquiry.",
        "urgency_reason": "Targeted for next quarter."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0034",
        "ticket_text": "Our marketing team noticed a typo in the footer of your public documentation site where 'privacy' is spelled incorrectly.",
        "category": "Technical Issue", "impact": "I1", "urgency": "U1",
        "scenario_type": "public_site_typo_report",
        "impact_reason": "Minor aesthetic typo on public website.",
        "urgency_reason": "Routine low-priority documentation feedback."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0035",
        "ticket_text": "Can I receive an itemized receipt for our annual subscription billed last month showing the tax breakdown?",
        "category": "Billing", "impact": "I1", "urgency": "U2",
        "scenario_type": "routine_receipt_copy_request",
        "impact_reason": "Routine duplicate document request.",
        "urgency_reason": "Standard accounting document retrieval."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0036",
        "ticket_text": "Do you have any recommendations for thermal label printers that work smoothly with your shipping label generator?",
        "category": "General Inquiry", "impact": "I1", "urgency": "U1",
        "scenario_type": "hardware_recommendation_inquiry",
        "impact_reason": "Hardware compatibility advice.",
        "urgency_reason": "Informational research."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0037",
        "ticket_text": "I wanted to pass along positive feedback for your support representative Sarah who helped resolve our billing ticket yesterday.",
        "category": "General Inquiry", "impact": "I1", "urgency": "U1",
        "scenario_type": "positive_customer_feedback",
        "impact_reason": "Customer compliments with no issue to solve.",
        "urgency_reason": "Zero urgency."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0038",
        "ticket_text": "Could you add a feature in the future that allows sorting search results by date modified in ascending order?",
        "category": "General Inquiry", "impact": "I1", "urgency": "U1",
        "scenario_type": "feature_suggestion",
        "impact_reason": "Product roadmap suggestion.",
        "urgency_reason": "Future enhancement consideration."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0039",
        "ticket_text": "What carriers do you support for domestic shipping in Australia?",
        "category": "Delivery", "impact": "I1", "urgency": "U1",
        "scenario_type": "carrier_coverage_inquiry",
        "impact_reason": "General capability question.",
        "urgency_reason": "No time sensitivity."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0040",
        "ticket_text": "Can I configure two-factor authentication using a YubiKey hardware token instead of an authenticator app?",
        "category": "Account Access", "impact": "I1", "urgency": "U2",
        "scenario_type": "security_hardware_inquiry",
        "impact_reason": "Authentication method inquiry.",
        "urgency_reason": "User setting up security preferences."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0041",
        "ticket_text": "Is there a limit on how many webhook endpoints we can register under an organization account?",
        "category": "General Inquiry", "impact": "I1", "urgency": "U1",
        "scenario_type": "api_limits_inquiry",
        "impact_reason": "Developer documentation inquiry.",
        "urgency_reason": "Informational query during development."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0042",
        "ticket_text": "Please confirm the email address where automated renewal reminders are sent before our plan renews in December.",
        "category": "Billing", "impact": "I1", "urgency": "U1",
        "scenario_type": "notification_email_verification",
        "impact_reason": "Minor profile verification.",
        "urgency_reason": "Renewal is in December; ample advance time."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0043",
        "ticket_text": "Our IT department is conducting an annual vendor security survey. Where can we download your SOC 2 compliance summary report?",
        "category": "General Inquiry", "impact": "I2", "urgency": "U1",
        "scenario_type": "compliance_document_request",
        "impact_reason": "Corporate compliance audit request.",
        "urgency_reason": "Annual survey cycle with multi-week turnaround."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0044",
        "ticket_text": "How do I revoke an invitation sent to a contractor who never joined our project space?",
        "category": "Account Access", "impact": "I1", "urgency": "U1",
        "scenario_type": "workspace_housekeeping",
        "impact_reason": "Routine workspace cleanup.",
        "urgency_reason": "No pending security threat or urgency."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0045",
        "ticket_text": "I noticed the terms of service page was updated last month. Is there a changelog detailing what modified?",
        "category": "General Inquiry", "impact": "I1", "urgency": "U1",
        "scenario_type": "tos_changelog_inquiry",
        "impact_reason": "Legal documentation question.",
        "urgency_reason": "Informational research."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0046",
        "ticket_text": "Can we customize the invoice header logo to match our subsidiary brand identity?",
        "category": "Billing", "impact": "I1", "urgency": "U1",
        "scenario_type": "branding_customization_inquiry",
        "impact_reason": "Aesthetic invoice customization request.",
        "urgency_reason": "No operational urgency."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0047",
        "ticket_text": "Our team would like to request that the API documentation include Python code snippets alongside the existing cURL examples.",
        "category": "General Inquiry", "impact": "I1", "urgency": "U1",
        "scenario_type": "documentation_improvement_suggestion",
        "impact_reason": "Public developer documentation suggestion.",
        "urgency_reason": "Feedback for future developer portal updates."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0048",
        "ticket_text": "Does your Android app support biometric fingerprint unlock on Samsung Galaxy devices?",
        "category": "Technical Issue", "impact": "I1", "urgency": "U1",
        "scenario_type": "device_compatibility_inquiry",
        "impact_reason": "Single user device capability question.",
        "urgency_reason": "Routine customer inquiry."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0049",
        "ticket_text": "We would like to add our corporate purchase order number to next month's invoice before payment is remitted.",
        "category": "Billing", "impact": "I1", "urgency": "U2",
        "scenario_type": "po_number_addition",
        "impact_reason": "Individual invoice metadata modification.",
        "urgency_reason": "Targeted for next month's invoice."
    },
    {
        "ticket_id": "TS-PRI-EXP2-0050",
        "ticket_text": "Can you provide the VAT registration number for your European operating entity for our procurement records?",
        "category": "Billing", "impact": "I1", "urgency": "U1",
        "scenario_type": "corporate_tax_id_request",
        "impact_reason": "Standard vendor onboarding document request.",
        "urgency_reason": "Routine procurement onboarding process."
    }
]

def generate_and_validate():
    print("="*80)
    print("GENERATING & VALIDATING PRIORITY EXPANSION DATASET V2")
    print("="*80)

    rows = []
    for item in SYNTHETIC_RECORDS:
        imp = item["impact"]
        urg = item["urgency"]
        # Derive authoritative priority deterministically
        prio = AUTHORITATIVE_RUBRIC.get((imp, urg))
        assert prio is not None, f"Invalid rubric pair: ({imp}, {urg})"
        
        row = {
            "ticket_id": item["ticket_id"],
            "ticket_text": item["ticket_text"],
            "category": item["category"],
            "impact": imp,
            "urgency": urg,
            "priority": prio,
            "priority_labels": f"{imp}|{urg}",
            "impact_reason": item["impact_reason"],
            "urgency_reason": item["urgency_reason"],
            "priority_reason": f"Authoritative rubric matrix ({imp} x {urg}) maps deterministically to {prio}.",
            "scenario_type": item["scenario_type"],
            "source_dataset": "ticketsense_priority_synthetic_v2",
            "text_origin": "synthetic_expansion_v2",
            "group_id": f"grp_{item['ticket_id']}",
            "split": "train"
        }
        rows.append(row)

    df_exp = pd.DataFrame(rows)
    print(f"Generated {len(df_exp)} synthetic records.")
    print("Priority distribution:\n", df_exp["priority"].value_counts().to_dict())
    print("Impact distribution:\n", df_exp["impact"].value_counts().to_dict())
    print("Urgency distribution:\n", df_exp["urgency"].value_counts().to_dict())

    # Leakage and Quality checks
    print("\n--- Running Leakage & Quality Validation ---")
    
    # 1. Duplicates within synthetic set
    dup_synth = df_exp["ticket_text"].duplicated().sum()
    assert dup_synth == 0, f"Found {dup_synth} duplicate texts in synthetic expansion."
    print("[PASS] Zero internal duplicate texts.")

    # 2. Check against protected datasets
    base_path = PROJECT_ROOT / "data" / "processed" / "ticketsense_dataset.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "ticketsense_test.csv"
    
    base_df = pd.read_csv(base_path)
    test_df = pd.read_csv(test_path)
    
    val_texts = set(base_df[base_df["split"] == "validation"]["ticket_text"].str.strip().str.lower())
    test_texts = set(test_df["ticket_text"].str.strip().str.lower())
    synth_texts = set(df_exp["ticket_text"].str.strip().str.lower())
    
    overlap_val = synth_texts.intersection(val_texts)
    overlap_test = synth_texts.intersection(test_texts)
    
    assert len(overlap_val) == 0, f"LEAKAGE: {len(overlap_val)} synthetic tickets overlap with validation set!"
    assert len(overlap_test) == 0, f"LEAKAGE: {len(overlap_test)} synthetic tickets overlap with locked test set!"
    print(f"[PASS] Zero overlap with validation set (0/{len(val_texts)}).")
    print(f"[PASS] Zero overlap with locked test set (0/{len(test_texts)}).")

    # Save to data/processed
    out_path = PROJECT_ROOT / "data" / "processed" / "priority_synthetic_expansion_v2.csv"
    df_exp.to_csv(out_path, index=False)
    print(f"[SUCCESS] Saved validated dataset to: {out_path}")

if __name__ == "__main__":
    generate_and_validate()
