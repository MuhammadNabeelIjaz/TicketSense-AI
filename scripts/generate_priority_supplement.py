"""Generate high-quality, realistic supplementary training data for Priority Classification.

Strictly adheres to the project's authoritative Impact x Urgency rubric:
  I1 x U1 -> Low
  I1 x U2 -> Low
  I1 x U3 -> Medium
  I2 x U1 -> Low
  I2 x U2 -> Medium
  I2 x U3 -> High
  I3 x U1 -> Medium
  I3 x U2 -> High
  I3 x U3 -> High

Generates realistic, varied support ticket texts across all 5 operational categories:
  - Account Access
  - Billing
  - Delivery
  - General Inquiry
  - Technical Issue

Includes realistic syntactic variations, natural language negation, differing severity,
and multi-sentence context.
Saves to data/processed/priority_supplementary_training.csv.
Leaves all original datasets immutable.
"""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "priority_supplementary_training.csv"

RUBRIC = {
    ("I1", "U1"): "Low",
    ("I1", "U2"): "Low",
    ("I1", "U3"): "Medium",
    ("I2", "U1"): "Low",
    ("I2", "U2"): "Medium",
    ("I2", "U3"): "High",
    ("I3", "U1"): "Medium",
    ("I3", "U2"): "High",
    ("I3", "U3"): "High",
}

# Carefully crafted realistic records across categories and underrepresented rubric cells
RECORDS = [
    # -------------------------------------------------------------
    # CELL: I3 | U3 -> High (High Impact, High Urgency)
    # Severe disruption, widespread failure, critical security/financial threat
    # -------------------------------------------------------------
    {
        "ticket_text": "Production API cluster is returning 503 errors across all regions. Our entire checkout pipeline is down and customers cannot complete purchases. Immediate emergency escalation required.",
        "category": "Technical Issue",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "Major security incident: our primary admin credentials were compromised and we see unauthorized exports of customer databases occurring right now. Lock down all administrative access immediately.",
        "category": "Account Access",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "Over 200 fraudulent debits hit our corporate credit line this morning totaling $150,000. Stop all outgoing automated transactions and reverse these unauthorized debits immediately.",
        "category": "Billing",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "Our entire warehouse dispatch software has crashed. 5,000 time-critical medical supply shipments scheduled for same-day delivery are frozen at the depot. Need urgent technical intervention now.",
        "category": "Delivery",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "Single Sign-On authentication is down for the entire company. All 1,200 staff members are locked out of enterprise applications and operations have halted. P0 outage.",
        "category": "Account Access",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "The latest production build corrupted the core customer database table. Transactions are failing and reporting queries are timing out. We need emergency database rollback assistance right now.",
        "category": "Technical Issue",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "Our billing integration charged our entire enterprise client base twice this billing cycle. Thousands of client accounts are impacted and banks are issuing merchant chargebacks. Halt the billing engine immediately.",
        "category": "Billing",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "Complete system outage: the web application returns a blank screen and 500 internal server error for all users nationwide. Customer support lines are flooded.",
        "category": "Technical Issue",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "All courier dispatch webhooks are failing with connection refused errors. Tracking updates and manifest exchanges for 15 partner carriers are stalled globally.",
        "category": "Delivery",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "Active brute-force attack detected on our central authentication endpoint. Thousands of suspicious login attempts per second are overwhelming LDAP services. Emergency firewall block needed.",
        "category": "Account Access",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "Payment gateway webhook failure is causing thousands of successful Stripe charges to fail order fulfillment on our platform. Urgent technical and financial intervention required.",
        "category": "Billing",
        "impact": "I3",
        "urgency": "U3",
    },
    {
        "ticket_text": "Critical data pipeline failure: incoming logistics data feeds from customs are failing schema validation, causing international clearance for all freight containers to be blocked at port.",
        "category": "Delivery",
        "impact": "I3",
        "urgency": "U3",
    },

    # -------------------------------------------------------------
    # CELL: I3 | U2 -> High (High Impact, Medium Urgency)
    # Organization-wide or customer-wide impact, but resolution needed in hours/today rather than seconds
    # -------------------------------------------------------------
    {
        "ticket_text": "The monthly financial reconciliation report generation times out for all finance team members. We cannot finalize month-end book closing scheduled for this afternoon.",
        "category": "Technical Issue",
        "impact": "I3",
        "urgency": "U2",
    },
    {
        "ticket_text": "A platform update has broken SAML role-mapping for all regional office managers. They cannot review or approve pending team workflows today. Please investigate today.",
        "category": "Account Access",
        "impact": "I3",
        "urgency": "U2",
    },
    {
        "ticket_text": "Our automated tax deduction calculations for enterprise invoices are calculating 0% VAT across all European customers. This needs correction before tomorrow's invoice run.",
        "category": "Billing",
        "impact": "I3",
        "urgency": "U2",
    },
    {
        "ticket_text": "Bulk label printing service is failing for all warehouse packing stations. Workers can still pack items manually, but shipping throughput is cut in half. Please resolve during this shift.",
        "category": "Delivery",
        "impact": "I3",
        "urgency": "U2",
    },
    {
        "ticket_text": "The corporate portal password expiration policy forced a reset on 400 user accounts, but the email delivery of reset tokens is queued with a 45-minute delay.",
        "category": "Account Access",
        "impact": "I3",
        "urgency": "U2",
    },
    {
        "ticket_text": "Analytics dashboard export functionality is broken for all enterprise accounts. PDF exports download as corrupt 0-byte files. Our board presentation is tomorrow morning.",
        "category": "Technical Issue",
        "impact": "I3",
        "urgency": "U2",
    },
    {
        "ticket_text": "Subscription renewals for the APAC subsidiary branch were invoiced in USD instead of AUD across 120 accounts. Please correct the currency mapping before funds are drafted.",
        "category": "Billing",
        "impact": "I3",
        "urgency": "U2",
    },
    {
        "ticket_text": "Customs documentation PDFs are missing the HS tariff code for all export shipments leaving the central depot tomorrow. Carrier will reject the batch if not fixed today.",
        "category": "Delivery",
        "impact": "I3",
        "urgency": "U2",
    },
    {
        "ticket_text": "Search indexing is stalled across the entire customer support knowledge base. Customers and staff are receiving empty search results for all documentation queries.",
        "category": "Technical Issue",
        "impact": "I3",
        "urgency": "U2",
    },
    {
        "ticket_text": "Two-factor authentication SMS gateway is intermittently failing for international numbers across EMEA. Several department heads are delayed logging in this morning.",
        "category": "Account Access",
        "impact": "I3",
        "urgency": "U2",
    },

    # -------------------------------------------------------------
    # CELL: I2 | U3 -> High (Medium Impact, High Urgency)
    # Localized impact (single key user/order), but immediate operational urgency
    # -------------------------------------------------------------
    {
        "ticket_text": "I am locked out of my corporate executive account right before an investor conference call. Password reset link is not arriving. Need my account unlocked immediately.",
        "category": "Account Access",
        "impact": "I2",
        "urgency": "U3",
    },
    {
        "ticket_text": "My corporate purchasing card was charged $1,200 for an order I never placed. I need this transaction blocked and refunded immediately before our ledger closes today.",
        "category": "Billing",
        "impact": "I2",
        "urgency": "U3",
    },
    {
        "ticket_text": "My urgent shipment containing trade show booth collateral is marked as delayed at sorting hub. The exhibition starts tomorrow morning at 8 AM. Need urgent reroute or courier intervention.",
        "category": "Delivery",
        "impact": "I2",
        "urgency": "U3",
    },
    {
        "ticket_text": "The payroll export tool throws an unhandled server error 500 whenever I click Export. Today is direct deposit submission cutoff and payroll cannot be submitted until this is fixed.",
        "category": "Technical Issue",
        "impact": "I2",
        "urgency": "U3",
    },
    {
        "ticket_text": "My account was locked after three incorrect PIN entries. I have client presentations starting in 30 minutes and cannot access any project files. Please unlock ASAP.",
        "category": "Account Access",
        "impact": "I2",
        "urgency": "U3",
    },
    {
        "ticket_text": "An unexpected charge of $850 was debited from my checking account causing an overdraft. Reverse this erroneous debit immediately so I do not incur additional bank penalties.",
        "category": "Billing",
        "impact": "I2",
        "urgency": "U3",
    },
    {
        "ticket_text": "My express delivery TS-7721 containing critical surgical replacement parts has been stalled in transit for 48 hours. Hospital surgery is scheduled for tomorrow. Need immediate tracer.",
        "category": "Delivery",
        "impact": "I2",
        "urgency": "U3",
    },
    {
        "ticket_text": "Production reporting dashboard crashes on load for my team lead account with error code 502 Bad Gateway. Client review meeting in 45 minutes.",
        "category": "Technical Issue",
        "impact": "I2",
        "urgency": "U3",
    },
    {
        "ticket_text": "I lost access to my authenticator app and cannot authenticate into my account. I have time-critical contract signing due in two hours.",
        "category": "Account Access",
        "impact": "I2",
        "urgency": "U3",
    },
    {
        "ticket_text": "My order was delivered to the wrong street address and signed by an unknown person. I need the courier to retrieve the package or dispatch an emergency replacement today.",
        "category": "Delivery",
        "impact": "I2",
        "urgency": "U3",
    },

    # -------------------------------------------------------------
    # CELL: I1 | U3 -> Medium (Low Impact, High Urgency)
    # Minor informational or routine matter, but customer demands urgent response
    # -------------------------------------------------------------
    {
        "ticket_text": "Quick question: what are your weekend technical support hours? I need to know in the next 10 minutes before I submit our on-call rotation schedule.",
        "category": "General Inquiry",
        "impact": "I1",
        "urgency": "U3",
    },
    {
        "ticket_text": "Can someone immediately verify if your software supports macOS Sequoia? I am standing in the electronics store right now deciding on a purchase.",
        "category": "General Inquiry",
        "impact": "I1",
        "urgency": "U3",
    },
    {
        "ticket_text": "I need a copy of my last paid receipt emailed to me within the next hour for my travel expense filing before the portal closes.",
        "category": "Billing",
        "impact": "I1",
        "urgency": "U3",
    },
    {
        "ticket_text": "Could you tell me immediately what the maximum parcel dimensions are for standard courier collection? The driver is arriving shortly.",
        "category": "Delivery",
        "impact": "I1",
        "urgency": "U3",
    },
    {
        "ticket_text": "Where is the button to change my display theme to dark mode? Please reply urgently, the bright screen is straining my eyes during a live webinar.",
        "category": "Technical Issue",
        "impact": "I1",
        "urgency": "U3",
    },
    {
        "ticket_text": "I need immediate confirmation of whether your enterprise tier includes telephone support. Our procurement team is finalizing vendor selection today.",
        "category": "General Inquiry",
        "impact": "I1",
        "urgency": "U3",
    },
    {
        "ticket_text": "Please confirm the spelling of the corporate billing name on our subscription account immediately for our auditor.",
        "category": "Billing",
        "impact": "I1",
        "urgency": "U3",
    },
    {
        "ticket_text": "What is the direct phone number for the regional dispatch office? I need to speak to the local supervisor right now.",
        "category": "General Inquiry",
        "impact": "I1",
        "urgency": "U3",
    },

    # -------------------------------------------------------------
    # CELL: I1 | U2 -> Low (Low Impact, Medium Urgency)
    # Minor informational or routine matter with standard timeline
    # -------------------------------------------------------------
    {
        "ticket_text": "Could you provide information on educational discounts for non-profit organizations? We are reviewing our budget for next month.",
        "category": "General Inquiry",
        "impact": "I1",
        "urgency": "U2",
    },
    {
        "ticket_text": "I would like to know if there is an option to change the default delivery courier for my future orders. Please let me know by this Friday.",
        "category": "Delivery",
        "impact": "I1",
        "urgency": "U2",
    },
    {
        "ticket_text": "Please provide clarification on how sales tax is calculated on digital service subscriptions in Canada. An answer this week would be appreciated.",
        "category": "Billing",
        "impact": "I1",
        "urgency": "U2",
    },
    {
        "ticket_text": "Is there a roadmap for adding dark mode to the mobile companion app? Just checking in for our team's upcoming hardware refresh.",
        "category": "General Inquiry",
        "impact": "I1",
        "urgency": "U2",
    },
    {
        "ticket_text": "I noticed a minor typo in the FAQ section regarding password length requirements. Passing this along for your documentation team to review.",
        "category": "General Inquiry",
        "impact": "I1",
        "urgency": "U2",
    },
    {
        "ticket_text": "Can you advise on the recommended screen resolution for the desktop client? Planning to purchase external monitors next week.",
        "category": "General Inquiry",
        "impact": "I1",
        "urgency": "U2",
    },
    {
        "ticket_text": "Where can I view the changelog for the latest software version 3.2? Need it for our monthly team release review.",
        "category": "Technical Issue",
        "impact": "I1",
        "urgency": "U2",
    },
    {
        "ticket_text": "Would it be possible to add our company tax ID number to past invoices from January? No rush, just before tax season.",
        "category": "Billing",
        "impact": "I1",
        "urgency": "U2",
    },

    # -------------------------------------------------------------
    # CELL: I3 | U1 -> Medium (High Impact, Low Urgency)
    # Organization-wide or platform-wide scope, but planned or long-term inquiry without time pressure
    # -------------------------------------------------------------
    {
        "ticket_text": "Our enterprise IT organization is planning a migration of 2,000 users to Okta SAML next quarter. Can you share technical documentation on automated user provisioning?",
        "category": "Account Access",
        "impact": "I3",
        "urgency": "U1",
    },
    {
        "ticket_text": "We are conducting an annual compliance audit of all third-party software vendors. Please provide your SOC 2 Type II report and data encryption specifications when available.",
        "category": "General Inquiry",
        "impact": "I3",
        "urgency": "U1",
    },
    {
        "ticket_text": "Our company is evaluating upgrading all 50 department accounts to the enterprise annual billing plan starting next fiscal year. What volume discount structures are offered?",
        "category": "Billing",
        "impact": "I3",
        "urgency": "U1",
    },
    {
        "ticket_text": "We are reviewing global freight logistics options for our upcoming international expansion next year. Does your API support multi-carrier dispatch in Latin America?",
        "category": "Delivery",
        "impact": "I3",
        "urgency": "U1",
    },
    {
        "ticket_text": "We are preparing for disaster recovery compliance testing next month. What are the documented RTO and RPO metrics for your multi-tenant database clusters?",
        "category": "General Inquiry",
        "impact": "I3",
        "urgency": "U1",
    },
    {
        "ticket_text": "Our security department requires all enterprise vendors to complete a vendor risk assessment questionnaire. There is no immediate deadline, please return by end of quarter.",
        "category": "General Inquiry",
        "impact": "I3",
        "urgency": "U1",
    },
    {
        "ticket_text": "We plan to integrate your webhook notifications into our internal Slack channels for all corporate accounts. Are there rate limits on outbound webhook delivery?",
        "category": "Technical Issue",
        "impact": "I3",
        "urgency": "U1",
    },
    {
        "ticket_text": "Can our accounting team receive consolidated annual billing statements for all our global operating entities? Looking ahead to next fiscal year planning.",
        "category": "Billing",
        "impact": "I3",
        "urgency": "U1",
    },

    # -------------------------------------------------------------
    # CELL: I2 | U2 -> Medium (Medium Impact, Medium Urgency)
    # Moderate individual or departmental disruption with standard turnaround
    # -------------------------------------------------------------
    {
        "ticket_text": "Whenever I try to export data to CSV, the file downloads with blank columns for customer addresses. This affects our weekly reporting workflow. Please investigate.",
        "category": "Technical Issue",
        "impact": "I2",
        "urgency": "U2",
    },
    {
        "ticket_text": "I requested a password reset yesterday and received the link, but clicking it displays an error saying token invalid. I need access restored so I can continue working.",
        "category": "Account Access",
        "impact": "I2",
        "urgency": "U2",
    },
    {
        "ticket_text": "Our package tracking number has shown in transit for four business days without a sorting facility scan. Could you check with the depot regarding the status?",
        "category": "Delivery",
        "impact": "I2",
        "urgency": "U2",
    },
    {
        "ticket_text": "I was billed twice for my monthly subscription on November 1st. Please review the duplicate transaction and apply a credit to my account.",
        "category": "Billing",
        "impact": "I2",
        "urgency": "U2",
    },
    {
        "ticket_text": "The web application is freezing whenever I upload PDF attachments larger than 5 MB. Other file formats upload without problem. Please check what is causing the freeze.",
        "category": "Technical Issue",
        "impact": "I2",
        "urgency": "U2",
    },
    {
        "ticket_text": "My colleague and I share access to our department workspace, but their invitations to new project boards keep failing with an authorization notice.",
        "category": "Account Access",
        "impact": "I2",
        "urgency": "U2",
    },
    {
        "ticket_text": "My package arrived with the outer carton opened and two of the five items missing. Please advise on how to file a partial loss claim.",
        "category": "Delivery",
        "impact": "I2",
        "urgency": "U2",
    },
    {
        "ticket_text": "Our annual subscription renewal invoice does not reflect our non-profit 20% discount code. Please reissue the invoice with the correct amount before payment.",
        "category": "Billing",
        "impact": "I2",
        "urgency": "U2",
    },

    # -------------------------------------------------------------
    # NEGATION AND AMBIGUITY CHALLENGE RECORDS:
    # Ensuring models do not trigger on keywords in negated contexts
    # -------------------------------------------------------------
    {
        "ticket_text": "This is not an emergency and our servers are not down, but we noticed a minor rendering issue with the chart legends on the dashboard.",
        "category": "Technical Issue",
        "impact": "I1",
        "urgency": "U1",
    },
    {
        "ticket_text": "No fraud occurred on my account and my card was not stolen, but I want to clarify how foreign currency transaction fees are listed on statements.",
        "category": "Billing",
        "impact": "I1",
        "urgency": "U1",
    },
    {
        "ticket_text": "I am not locked out of my account and I have full access, but I wanted to ask where the setting is to enable two-factor authentication.",
        "category": "Account Access",
        "impact": "I1",
        "urgency": "U1",
    },
    {
        "ticket_text": "There is no delivery delay with my parcel, it arrived on time, but I wanted to leave positive feedback for the polite delivery driver.",
        "category": "General Inquiry",
        "impact": "I1",
        "urgency": "U1",
    },
    {
        "ticket_text": "We are not experiencing any outage today. We just wanted to confirm your standard SLA response times for future reference.",
        "category": "General Inquiry",
        "impact": "I1",
        "urgency": "U1",
    },
    {
        "ticket_text": "I did not receive an unauthorized charge. However, could you explain the prorated adjustment on line 3 of my invoice?",
        "category": "Billing",
        "impact": "I1",
        "urgency": "U1",
    },
]

# Build DataFrame and validate
rows = []
for idx, r in enumerate(RECORDS, 1):
    imp = r["impact"]
    urg = r["urgency"]
    prio = RUBRIC[(imp, urg)]
    priority_label = f"{imp}|{urg}"
    ticket_id = f"TS-PRI-SUPP-{idx:04d}"

    rows.append({
        "ticket_id": ticket_id,
        "ticket_text": r["ticket_text"],
        "category": r["category"],
        "priority": prio,
        "source_dataset": "ticketsense_priority_supplement_v1",
        "original_label": r["category"],
        "source_record_id": ticket_id,
        "text_origin": "synthetic_priority_expansion",
        "group_id": f"grp_{ticket_id}",
        "split": "train",
        "priority_labels": priority_label,
        "priority_reason": f"Supplementary training record addressing underrepresented {priority_label} ({prio}) cell under authoritative rubric."
    })

supp_df = pd.DataFrame(rows)

# Strict validation
print(f"Total supplementary records generated: {len(supp_df)}")
print("\nBreakdown by Impact:")
print(supp_df["impact"] if "impact" in supp_df else supp_df["priority_labels"].apply(lambda x: x.split("|")[0]).value_counts())
print("\nBreakdown by Urgency:")
print(supp_df["priority_labels"].apply(lambda x: x.split("|")[1]).value_counts())
print("\nBreakdown by Priority:")
print(supp_df["priority"].value_counts())
print("\nBreakdown by Category:")
print(supp_df["category"].value_counts())

# Validate 100% adherence to rubric
for _, row in supp_df.iterrows():
    imp, urg = row["priority_labels"].split("|")
    expected_prio = RUBRIC[(imp, urg)]
    assert row["priority"] == expected_prio, f"Rubric violation in {row['ticket_id']}: {imp}x{urg} gave {row['priority']} instead of {expected_prio}"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
supp_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
print(f"\nSaved validated supplementary dataset to: {OUTPUT_PATH}")
