"""Script to generate high-quality, diverse synthetic support tickets for underrepresented scenarios."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from ticketsense.config import PROCESSED_DATA_DIR

SYNTHETIC_DATA_PATH = PROCESSED_DATA_DIR / "ticketsense_synthetic_expansion_v1.csv"

# 40 targeted tickets per category = 200 total
SYNTHETIC_RECORDS = [
    # =========================================================================
    # 1. ACCOUNT ACCESS (40 records)
    # Scenarios: Magic links, 2FA recovery, SSO/SAML, concurrent sessions,
    # email change, account reactivation, role access, typos, short/long.
    # =========================================================================
    ("Account Access", "I requested a magic link to sign in, but clicking it just loops me back to the login screen without authenticating.", "Medium"),
    ("Account Access", "My old phone was damaged so I cannot get my 2FA authentication codes. How do I reset my two-factor device?", "High"),
    ("Account Access", "Our company single sign-on via Okta is throwing a SAML assertion error: 'token expired or signature invalid'.", "High"),
    ("Account Access", "Whenever I log into the mobile app, my active web session gets terminated immediately. Can I stay signed in on both?", "Low"),
    ("Account Access", "I need to update the primary account holder email address from admin_old@company.com to admin_new@company.com.", "Medium"),
    ("Account Access", "My account seems to have been archived due to 90 days of inactivity. Can an administrator reactivate my profile?", "Medium"),
    ("Account Access", "cant log in to my accoont. getting invalid credentails error even though I just changed my pasword five minutes ago.", "Medium"),
    ("Account Access", "Password reset email is not arriving in my inbox or spam folder after multiple requests.", "High"),
    ("Account Access", "I entered my password wrong three times and now my IP is blocked for 24 hours. Please unblock my IP address.", "High"),
    ("Account Access", "I am the new IT director and need super-admin ownership transferred to my corporate profile.", "Medium"),
    ("Account Access", "The 6-digit verification SMS code is taking over 20 minutes to arrive, by which time the web page has timed out.", "High"),
    ("Account Access", "We have federated Google Workspace login enabled, but new team members receive 'user not provisioned in tenant'.", "High"),
    ("Account Access", "I forgot my security questions and cannot complete the identity verification step to regain access.", "Medium"),
    ("Account Access", "Logged out unexpectedly during an active workflow and now the portal says 'Account status suspended pending verification'.", "High"),
    ("Account Access", "Can you help me revoke access for an employee who left our organization yesterday?", "High"),
    ("Account Access", "The authenticator app backup codes I saved when setting up my account are being rejected as invalid.", "High"),
    ("Account Access", "I keep getting an 'expired session' prompt every two minutes while trying to edit my user profile.", "Low"),
    ("Account Access", "Locked out of my dashboard on a Saturday morning. I have an urgent client presentation and need access immediately.", "High"),
    ("Account Access", "Please unlock user account jsmith@enterprise.org after repeated failed login attempts.", "Medium"),
    ("Account Access", "Need to force logout on all active devices because I suspect someone unauthorized accessed my profile.", "High"),
    ("Account Access", "The biometric Touch ID login prompt fails silently and will not fall back to the master password form.", "Medium"),
    ("Account Access", "Cannot access the admin console. I was granted administrator role yesterday but my dashboard still shows viewer permissions.", "Medium"),
    ("Account Access", "Trying to sign in with Microsoft Azure AD SSO but getting AADSTS50011 redirect URI mismatch.", "High"),
    ("Account Access", "My account was created by our company administrator, but the invitation link says 'Invitation token has expired'.", "Medium"),
    ("Account Access", "I need to reset the secondary email address associated with password recovery.", "Low"),
    ("Account Access", "When attempting to log in, I get redirected in an infinite loop between auth.domain.com and the home dashboard.", "High"),
    ("Account Access", "Can you clear my remembered trusted devices so that two-factor authentication is required again on next sign-in?", "Low"),
    ("Account Access", "Error: 'Your account has been temporarily disabled by a security policy'. What does this mean?", "High"),
    ("Account Access", "I lost access to my corporate domain email and need to switch my login to my personal email to recover my portfolio.", "High"),
    ("Account Access", "Every time I click the password reset link from my phone, it says 'invalid hash or token already used'.", "Medium"),
    ("Account Access", "Need assistance enabling hardware security key (YubiKey / WebAuthn) for our team login.", "Low"),
    ("Account Access", "Account lockout notice received via email even though I have not tried logging in today.", "High"),
    ("Account Access", "How can I merge two accounts that were registered under slightly different email aliases?", "Low"),
    ("Account Access", "We need to re-verify our business domain for Google SSO authentication before our grace period expires.", "Medium"),
    ("Account Access", "Unable to log into the staging environment with production credentials.", "Low"),
    ("Account Access", "Is there a way to bypass 2FA temporarily? My registered mobile device was stolen while traveling abroad.", "High"),
    ("Account Access", "The CAPTCHA on the login screen refuses to load, preventing me from submitting my username and password.", "Medium"),
    ("Account Access", "I am unable to switch between multiple organization accounts linked to my primary email login.", "Medium"),
    ("Account Access", "Received error 'User profile corrupt or missing identity attributes' upon logging in this morning.", "High"),
    ("Account Access", "Please assist with unlocking my account; I have submitted proof of identification to the security desk.", "Medium"),

    # =========================================================================
    # 2. BILLING (40 records)
    # Scenarios: Overage caps, undeliverable refunds, tax exemption/VAT,
    # prorated items, payment retry/grace period, FX fees, invoice disputes.
    # =========================================================================
    ("Billing", "We would like to configure a hard spending cap on our monthly API usage to avoid unexpected overage charges.", "Medium"),
    ("Billing", "My refund was returned by my banking institution because the account was closed. Please reissue the payout to my new IBAN.", "High"),
    ("Billing", "Our tax department needs an updated invoice showing our valid EU VAT number (DE318920192) and reverse charge notes.", "Medium"),
    ("Billing", "I was billed for 15 user seats this month, but our team only had 10 active seats configured in the admin console.", "Medium"),
    ("Billing", "The automated credit card charge failed because our company corporate card was replaced. How do we update card details without service disruption?", "High"),
    ("Billing", "Why did my monthly subscription jump from $29 to $49 without any advance notification or plan change confirmation?", "Medium"),
    ("Billing", "Can you explain the prorated calculation on invoice #INV-88219? The math does not appear to reflect our mid-cycle downgrade.", "Low"),
    ("Billing", "Please issue a formal credit note for invoice INV-1049 which was paid twice in error.", "Medium"),
    ("Billing", "We are a registered 501(c)(3) non-profit organization and need sales tax removed from all future monthly billings.", "Low"),
    ("Billing", "The invoice PDF download link in our account settings produces a corrupt file with zero bytes.", "Medium"),
    ("Billing", "I need to update the billing department contact email address for receiving monthly receipts.", "Low"),
    ("Billing", "Our account was charged an international currency conversion fee. Can we be billed directly in Euros instead of USD?", "Low"),
    ("Billing", "We downgraded to the free tier two weeks ago, but our card was just billed for another full annual cycle.", "High"),
    ("Billing", "Please send us a Certificate of Residence or W-9 form so our accounting department can process your payment.", "Low"),
    ("Billing", "Can we set up automated invoicing with Net 30 payment terms via wire transfer instead of credit card auto-debit?", "Medium"),
    ("Billing", "There is an unknown recurring fee of $12.50 labeled 'platform add-on' that nobody on our team authorized.", "High"),
    ("Billing", "We need a consolidated invoice covering all three sub-accounts under our parent enterprise agreement.", "Low"),
    ("Billing", "I cancelled my subscription within the 30-day money-back guarantee window. When will the refund post to my card?", "Medium"),
    ("Billing", "Our bank declined the transaction with error code 51 (insufficient funds). Can you re-run the transaction now that funds are available?", "Medium"),
    ("Billing", "Received a dunning email warning that our account will be suspended in 48 hours due to an unpaid balance.", "High"),
    ("Billing", "The billing summary dashboard does not match the actual amount debited from our bank account statement.", "High"),
    ("Billing", "Can you provide an itemized breakdown of our data transfer overage fees for the previous billing cycle?", "Low"),
    ("Billing", "We were double-charged on November 3rd for subscription plan TS-PRO. Please refund the duplicate transaction.", "High"),
    ("Billing", "I need our company purchase order number (PO #4500918) printed on all future invoices.", "Low"),
    ("Billing", "Is it possible to pause our subscription billing for two months during our seasonal shutdown?", "Low"),
    ("Billing", "My discount promo code 'WELCOME50' was not applied at checkout, and I was charged the full retail price.", "Medium"),
    ("Billing", "Where can I view and download past payment receipts from the last fiscal year?", "Low"),
    ("Billing", "Can you transfer the remaining credit balance from our terminated account to our new active organization?", "Medium"),
    ("Billing", "Our account incurred unexpected storage overage charges because deleted items remained in the retention cache.", "Medium"),
    ("Billing", "We need proof of payment and a zero-balance receipt for our annual audit compliance.", "Low"),
    ("Billing", "Why was state sales tax charged on our invoice when digital software is exempt in our jurisdiction?", "Low"),
    ("Billing", "Our annual contract renewed automatically, but we submitted our cancellation notice 30 days prior as required.", "High"),
    ("Billing", "Can we split our monthly bill across two separate corporate credit cards?", "Low"),
    ("Billing", "We require an invoice reissued with our full legal entity name rather than our trade name.", "Low"),
    ("Billing", "Refund requested two weeks ago has still not appeared on our bank statement. Please provide the Acquirer Reference Number (ARN).", "High"),
    ("Billing", "Payment gateway keeps rejecting our corporate Amex card with error 'generic decline'.", "Medium"),
    ("Billing", "We were charged for an enterprise feature trial that was advertised as 100% free with no credit card required.", "High"),
    ("Billing", "How do we switch our billing currency from GBP to EUR?", "Low"),
    ("Billing", "Please apply the $200 promotional credit promised by our account manager to invoice #9012.", "Medium"),
    ("Billing", "Our subscription shows 'past due' even though our bank confirms the wire transfer cleared on Monday.", "High"),

    # =========================================================================
    # 3. TECHNICAL ISSUE (40 records)
    # Scenarios: API rate limits (429), CPU/memory spikes, locale/number formats,
    # desktop push notifications, webhook failures, syntax/query bugs, 500 errors.
    # =========================================================================
    ("Technical Issue", "Our application is receiving HTTP 429 Too Many Requests on the very first API call of the day.", "High"),
    ("Technical Issue", "The backend worker process spikes CPU utilization to 100% and crashes whenever a CSV export exceeds 500 rows.", "High"),
    ("Technical Issue", "In German locale settings, decimal numbers with commas are incorrectly parsed, causing calculations to be off by 100x.", "Medium"),
    ("Technical Issue", "Desktop notifications are enabled in preferences, but no pop-up alerts or sound triggers appear on Windows 11.", "Medium"),
    ("Technical Issue", "Our webhook endpoint receives 502 Bad Gateway responses when your event dispatcher delivers webhook events.", "High"),
    ("Technical Issue", "Custom equality operator fails when comparing nested JSON attributes in the workflow builder.", "Medium"),
    ("Technical Issue", "Uploading any attachment larger than 8MB fails with an unhandled '500 Internal Server Error' from the API gateway.", "High"),
    ("Technical Issue", "The dashboard analytics charts render completely blank in Firefox 120, while working properly in Chrome.", "Medium"),
    ("Technical Issue", "Memory consumption steadily climbs by 200MB per hour until the Node.js process runs out of heap memory.", "High"),
    ("Technical Issue", "Special UTF-8 characters like German umlauts (ä, ö, ü) are exported as corrupted question marks in PDF reports.", "Medium"),
    ("Technical Issue", "Clicking 'Save Changes' on the integration settings tab triggers an endless spinning loader with no network response.", "Medium"),
    ("Technical Issue", "The iOS mobile application crashes immediately upon launch on iOS 17.2 devices.", "High"),
    ("Technical Issue", "Our scheduled data sync job failed overnight with error code 'ETIMEDOUT: Connection to database timed out'.", "High"),
    ("Technical Issue", "Date filters in the search query do not respect the user's configured time zone (UTC+5:00).", "Low"),
    ("Technical Issue", "Browser developer console shows CORS policy error: 'No Access-Control-Allow-Origin header is present on requested resource'.", "High"),
    ("Technical Issue", "Exporting tables to Excel (.xlsx) generates a damaged file that Excel reports as unreadable.", "Medium"),
    ("Technical Issue", "Drag-and-drop file upload zone becomes unresponsive after cancelling the first file picker dialog.", "Low"),
    ("Technical Issue", "WebSocket connection drops every 60 seconds with close code 1006 (abnormal closure).", "High"),
    ("Technical Issue", "Typeahead search autocomplete stops returning suggestions after typing more than four characters.", "Low"),
    ("Technical Issue", "Database migration script failed on step 042 with error: 'Column already exists in table users'.", "High"),
    ("Technical Issue", "The PDF preview modal displays an unformatted raw base64 string instead of rendering the document view.", "Medium"),
    ("Technical Issue", "Search query containing an apostrophe or single quote causes an SQL syntax error exception.", "High"),
    ("Technical Issue", "Bulk delete operation fails silently without deleting any records or providing an error notification.", "High"),
    ("Technical Issue", "The system displays time stamps in UTC instead of the configured local browser time zone.", "Low"),
    ("Technical Issue", "SSL handshake failed when connecting our custom domain to your CDN edge proxy.", "High"),
    ("Technical Issue", "Rich text editor removes HTML formatting and paragraphs when toggling between visual and markdown mode.", "Low"),
    ("Technical Issue", "Getting a 403 Forbidden error when trying to download our own audit log archive.", "Medium"),
    ("Technical Issue", "The report generator times out after 30 seconds when filtering across date ranges greater than 30 days.", "Medium"),
    ("Technical Issue", "Mobile responsive view hides the primary submit button below the fold on iPhone SE screens.", "Medium"),
    ("Technical Issue", "Kafka consumer group lag is growing continuously because messages are failing deserialization.", "High"),
    ("Technical Issue", "Two simultaneous edit requests on the same ticket cause a race condition and overwrite each other's notes.", "Medium"),
    ("Technical Issue", "The Chrome browser extension is throwing an unhandled runtime error in background.js on page navigation.", "Medium"),
    ("Technical Issue", "Table pagination shows 'Page 1 of 0' and the Next button is disabled even though 40 records exist.", "Low"),
    ("Technical Issue", "REST API endpoint /v1/tickets returns HTTP 503 Service Unavailable during peak morning traffic.", "High"),
    ("Technical Issue", "Dark mode theme causes black text on dark grey background, rendering the table unreadable.", "Low"),
    ("Technical Issue", "Automated email notifications contain broken image URLs and missing CSS styling.", "Low"),
    ("Technical Issue", "The live chat widget does not reconnect after the user switches between Wi-Fi and mobile data.", "Medium"),
    ("Technical Issue", "Export to CSV truncates leading zeroes from customer zip codes and telephone numbers.", "Low"),
    ("Technical Issue", "Background worker queue is backed up with 10,000 stalled jobs due to an unhandled null pointer exception.", "High"),
    ("Technical Issue", "Grid column resizing is broken; columns collapse to zero width upon mouse drag.", "Low"),

    # =========================================================================
    # 4. GENERAL INQUIRY (40 records)
    # Scenarios: OS compatibility (Linux/macOS), tutorials/guides, privacy/GDPR/SOC2,
    # feature roadmap, pricing/discounts, SLA guarantees, partner programs.
    # =========================================================================
    ("General Inquiry", "Does your desktop application support Ubuntu Linux 22.04 LTS, or is it limited to Windows and macOS?", "Low"),
    ("General Inquiry", "Where can we find comprehensive video tutorials and step-by-step user guides for onboarding new team members?", "Low"),
    ("General Inquiry", "How is customer personal data stored and encrypted at rest, and can we review your SOC 2 Type II compliance report?", "Low"),
    ("General Inquiry", "Do you offer discounted pricing tiers for educational institutions and registered non-profit organizations?", "Low"),
    ("General Inquiry", "What are the guaranteed uptime percentages and support response times outlined in your Enterprise Service Level Agreement (SLA)?", "Low"),
    ("General Inquiry", "Is there a public product roadmap where customers can vote on upcoming feature requests?", "Low"),
    ("General Inquiry", "Do you have an affiliate or solution partner program for system integrators and technology consultants?", "Low"),
    ("General Inquiry", "Can you explain the key feature differences between the Professional plan and the Enterprise tier?", "Low"),
    ("General Inquiry", "Are your cloud servers hosted within the European Union to comply with GDPR data sovereignty requirements?", "Low"),
    ("General Inquiry", "What are your standard technical support desk operating hours, and do you offer 24/7 emergency coverage?", "Low"),
    ("General Inquiry", "Do you provide hands-on product demonstrations or trial onboarding sessions for large engineering teams?", "Low"),
    ("General Inquiry", "Can we request a custom Data Processing Agreement (DPA) signed with our standard vendor contractual clauses?", "Low"),
    ("General Inquiry", "Is there a limit on how many webhook subscriptions an organization can configure under a single tenant?", "Low"),
    ("General Inquiry", "What programming languages and SDKs are officially supported for your public REST API?", "Low"),
    ("General Inquiry", "Do you support HIPAA compliance and will your company sign a Business Associate Agreement (BAA)?", "Low"),
    ("General Inquiry", "Can you provide references or case studies from other healthcare companies currently utilizing your platform?", "Low"),
    ("General Inquiry", "What is the maximum file attachment size supported for customer tickets and document uploads?", "Low"),
    ("General Inquiry", "How frequently do you release product updates and scheduled maintenance patches?", "Low"),
    ("General Inquiry", "Is there an on-premise or self-hosted deployment option available for defense and financial clients?", "Low"),
    ("General Inquiry", "What browser versions are officially supported by your web portal interface?", "Low"),
    ("General Inquiry", "Can we export all of our historical ticket and customer data if we decide to discontinue our contract?", "Low"),
    ("General Inquiry", "Are there any usage quotas or query limits on the standard search indexing engine?", "Low"),
    ("General Inquiry", "Do you offer multi-year contract discounts if paid upfront for 3 years?", "Low"),
    ("General Inquiry", "Where can I sign up for the developer beta program to test upcoming API endpoints?", "Low"),
    ("General Inquiry", "Can multiple administrators share a single team dashboard or are named seat licenses required for each user?", "Low"),
    ("General Inquiry", "What encryption algorithms and TLS cipher suites are utilized for data in transit across your API endpoints?", "Low"),
    ("General Inquiry", "Are there native integration plugins available for Jira, Salesforce, and Zendesk?", "Low"),
    ("General Inquiry", "How do we request a feature enhancement for our specific organizational workflow?", "Low"),
    ("General Inquiry", "Do you have formal business continuity and disaster recovery policies that our audit team can inspect?", "Low"),
    ("General Inquiry", "What is the recommended hardware specification for hosting your local caching proxy?", "Low"),
    ("General Inquiry", "Are customer service calls and chat interactions recorded for internal training and quality assurance?", "Low"),
    ("General Inquiry", "Does your service comply with Accessibility guidelines (WCAG 2.1 AA standards)?", "Low"),
    ("General Inquiry", "Can we schedule a call with an enterprise solutions architect before purchasing an annual license?", "Low"),
    ("General Inquiry", "Is dark mode theme supported across all web views and mobile applications?", "Low"),
    ("General Inquiry", "What is the policy regarding inactive accounts and retention of archived customer data?", "Low"),
    ("General Inquiry", "Do you support single sign-on with Google Workspace on the Starter plan or is it exclusive to Enterprise?", "Low"),
    ("General Inquiry", "Can we customize the branding, logos, and portal domain name with a white-label package?", "Low"),
    ("General Inquiry", "How do we register as a certified implementation partner on your company marketplace?", "Low"),
    ("General Inquiry", "What happens to our project data when a free trial period expires?", "Low"),
    ("General Inquiry", "Are there user community forums or Discord channels where developers share tips and workflow automations?", "Low"),

    # =========================================================================
    # 5. DELIVERY (40 records)
    # Scenarios: Damaged in transit, signature required, mid-transit reroute,
    # customs hold, partial shipment, porch piracy / false delivery, stuck tracking.
    # =========================================================================
    ("Delivery", "My package arrived with the outer shipping box crushed and the glassware inside completely broken.", "High"),
    ("Delivery", "The delivery driver marked 'attempted delivery - signature required' but nobody rang our doorbell or knocked.", "Medium"),
    ("Delivery", "I accidentally entered my old apartment address on order #99214. Can you redirect the parcel to my new address while in transit?", "High"),
    ("Delivery", "Tracking shows my international parcel has been held in customs for 10 days awaiting import declaration documents.", "High"),
    ("Delivery", "I ordered three items under order #4491, but only received one item today. Where are the remaining two items?", "Medium"),
    ("Delivery", "Tracking status says 'Delivered to resident at front door' at 2:15 PM, but I was home and no parcel was delivered.", "High"),
    ("Delivery", "The tracking number provided has been stuck on 'Shipping label created, carrier awaiting package' for 8 days.", "Medium"),
    ("Delivery", "The parcel pickup locker PIN sent to my phone has expired and the automated locker will not release my order.", "Medium"),
    ("Delivery", "The delivery courier delivered my package to the wrong building down the street according to the delivery photo.", "High"),
    ("Delivery", "My package was supposed to arrive by guaranteed 2-day delivery on Friday, but it has not arrived and tracking is delayed.", "Medium"),
    ("Delivery", "Can I request that the courier hold my package at the local depot for pickup instead of attempting home delivery?", "Low"),
    ("Delivery", "Tracking says 'Return to sender - insufficient address information provided'. Can you reship this with the correct unit number?", "High"),
    ("Delivery", "The delivery driver left my expensive electronic parcel out in the pouring rain without protective plastic covering.", "Medium"),
    ("Delivery", "Carrier tracking states 'Delivery exception: animal interference / loose dog on premises' which is incorrect.", "Low"),
    ("Delivery", "I need to change the delivery date because our office will be closed for the national holiday on Thursday.", "Low"),
    ("Delivery", "The shipping address on my order confirmation email is missing my apartment unit number (Apt 4B).", "High"),
    ("Delivery", "Package has been bouncing back and forth between two regional sorting distribution facilities for five days.", "Medium"),
    ("Delivery", "Is Saturday delivery included with standard courier shipping or does it require an additional expedited surcharge?", "Low"),
    ("Delivery", "Courier claims the recipient refused delivery, but nobody from our household spoke with any courier.", "High"),
    ("Delivery", "Received a delivery notification text, but the parcel locker compartment was empty when I opened it.", "High"),
    ("Delivery", "How can I obtain proof of delivery documentation including the recipient's signature for an insurance claim?", "Medium"),
    ("Delivery", "The tracking barcode on my order cannot be found on the FedEx website. Has the order actually shipped?", "Medium"),
    ("Delivery", "My shipment arrived missing the power adapter and installation accessories from the sealed box.", "Medium"),
    ("Delivery", "Can you provide an estimated delivery time window so someone can be home to sign for the freight delivery?", "Low"),
    ("Delivery", "Package was marked as delivered yesterday, but our building concierge confirmed no packages were accepted for my unit.", "High"),
    ("Delivery", "Tracking status says 'Held at regional hub due to severe weather delays'. When will transit resume?", "Low"),
    ("Delivery", "My order was split into two shipments with different tracking numbers, but only one tracking link is active.", "Medium"),
    ("Delivery", "The delivery driver threw the parcel over our 6-foot fence onto the concrete patio, damaging the contents.", "High"),
    ("Delivery", "Need to update shipping speed from ground freight to next-day air before the warehouse packages the order.", "High"),
    ("Delivery", "Courier tracking reports 'Business closed' at 11:00 AM on a Tuesday, but our reception is open 9 to 5.", "Medium"),
    ("Delivery", "Why was my package routed through another state before heading towards my destination city?", "Low"),
    ("Delivery", "I received someone else's order! The shipping label has my address, but the packing slip and items belong to another customer.", "High"),
    ("Delivery", "Parcel was marked delivered to parcel locker #12, but the locker door is stuck and will not unlatch.", "Medium"),
    ("Delivery", "Carrier tracking indicates a delivery delay caused by mechanical breakdown of transport vehicle.", "Low"),
    ("Delivery", "Can I add special gate code instructions (Gate Code #4412) for the courier driver before delivery?", "Low"),
    ("Delivery", "Package has been in transit for over three weeks for a domestic order. Can you declare the package lost and issue a replacement?", "High"),
    ("Delivery", "Delivery driver left the package in our open apartment building lobby where anyone passing by could take it.", "Medium"),
    ("Delivery", "Do you deliver to military APO / FPO postal addresses or international P.O. boxes?", "Low"),
    ("Delivery", "The delivery confirmation picture clearly shows a completely different house and porch than mine.", "High"),
    ("Delivery", "My tracking number updates stopped abruptly four days ago after reaching the destination distribution center.", "Medium"),
]


def generate_dataset():
    records = []
    for idx, (cat, text, priority) in enumerate(SYNTHETIC_RECORDS, start=1):
        ticket_id = f"TS-SYN-{idx:04d}"
        records.append({
            "ticket_id": ticket_id,
            "ticket_text": text,
            "category": cat,
            "priority": priority,
            "source_dataset": "synthetic_expansion_v1",
            "original_label": cat.lower().replace(" ", "_"),
            "source_record_id": 1000 + idx,
            "text_origin": "synthetic_generated_validated",
            "group_id": "G-SYN",
            "split": "train",
            "priority_labels": "I2|U2" if priority == "High" else ("I2|U1" if priority == "Medium" else "I1|U1"),
            "priority_reason": f"Synthetic expansion record for underrepresented {cat} scenarios.",
        })

    df = pd.DataFrame(records)
    SYNTHETIC_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SYNTHETIC_DATA_PATH, index=False, encoding="utf-8")
    print(f"Generated {len(df)} synthetic records saved to: {SYNTHETIC_DATA_PATH}")
    print("Category breakdown:\n", df["category"].value_counts())
    print("Priority breakdown:\n", df["priority"].value_counts())


if __name__ == "__main__":
    generate_dataset()
