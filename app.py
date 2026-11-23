"""TicketSense-AI — Smart Customer Support Ticket Classifier.

Interactive Streamlit application providing:
1. Single customer message classification with real-time confidence & uncertainty alerts.
2. Batch CSV prediction, filtering, and export.
3. Auditable human correction logging to prevent historical dataset mutation.
4. Model performance & governance dashboard.
"""

import os
from pathlib import Path
import pandas as pd
import streamlit as st

from ticketsense.config import (
    APPROVED_CATEGORIES,
    DEFAULT_UNCERTAINTY_THRESHOLD,
    DEFAULT_UNCERTAINTY_MARGIN,
    PRIORITY_POLICY_STATUS_MESSAGE,
    RUNTIME_CORRECTIONS_PATH,
    REPORTS_DIR,
)
from ticketsense.inference.engine import InferenceEngine
from ticketsense.data.corrections import log_manual_correction, load_corrections_log


# Page configuration
st.set_page_config(
    page_title="TicketSense-AI",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_engine() -> InferenceEngine:
    """Cache inference engine instance across sessions."""
    return InferenceEngine()


def render_header():
    """Render top banner and governance status."""
    st.title("🎫 TicketSense-AI")
    st.markdown(
        "**Smart Customer Support Ticket Classifier** | "
        "*Production-grade text classification, uncertainty quantification, and human-in-the-loop review.*"
    )

    with st.expander("ℹ️ System Architecture & Governance Status", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "**Category Classifier:** Active  \n"
                "- Architecture: TF-IDF + Regularized Logistic Regression  \n"
                "- Validation Accuracy: **80.0%** | Macro F1: **0.800**  \n"
                "- Split: 400 Train, 100 Validation (100 Test isolated)"
            )
        with col2:
            st.markdown(
                "**Priority Classifier:** ⏸️ **PAUSED**  \n"
                f"- Policy Status: *{PRIORITY_POLICY_STATUS_MESSAGE}*  \n"
                "- Guardrail: Machine learning priority training is paused pending Gate G1 supervisor approval. Heuristic indicator provided for operational triage."
            )


def render_single_ticket_tab(engine: InferenceEngine):
    """Render single ticket classification interface."""
    st.subheader("Single Ticket Classification")

    # Sample tickets for one-click testing
    samples = {
        "Custom Message": "",
        "Billing: Unexpected recurring charge": "I was charged $49.99 on my credit card this morning for a subscription I cancelled last week. Please reverse this transaction immediately.",
        "Account Access: Password reset loop": "I am locked out of my corporate portal account and the password reset link redirects to an expired token page. Please assist.",
        "Delivery: Package tracking stuck": "My tracking number TS-9821 has been marked 'in transit' for eight days with no scan update. Can you verify if the courier lost the parcel?",
        "Technical Issue: 500 error on dashboard": "Whenever I click 'Export Report', the web application crashes with internal server error code 500. Console shows unhandled promise rejection.",
        "Ambiguous / Uncertain Edge Case": "My account was billed twice but I also cannot log in to download my invoice.",
    }

    # Session state initialization for persistence across widget reruns
    if "single_prediction" not in st.session_state:
        st.session_state.single_prediction = None
    if "single_ticket_text" not in st.session_state:
        st.session_state.single_ticket_text = ""

    selected_sample = st.selectbox(
        "Select a sample ticket or enter your own below:",
        options=list(samples.keys()),
        index=0,
    )

    initial_text = samples[selected_sample] if selected_sample != "Custom Message" else ""

    ticket_input = st.text_area(
        "Customer Support Message:",
        value=initial_text,
        height=140,
        placeholder="Type or paste incoming customer message here...",
        help="Input must contain at least 5 characters.",
    )

    col_btn, col_thresh = st.columns([1, 2])
    with col_thresh:
        uncertainty_thresh = st.slider(
            "Human Review Confidence Threshold:",
            min_value=0.50,
            max_value=0.90,
            value=DEFAULT_UNCERTAINTY_THRESHOLD,
            step=0.05,
            help="Predictions with confidence below this threshold are flagged for human supervisor review.",
        )
    with col_btn:
        st.write("")
        st.write("")
        predict_clicked = st.button("Classify Ticket", type="primary", use_container_width=True)

    if predict_clicked or (ticket_input and selected_sample != "Custom Message" and ticket_input != st.session_state.single_ticket_text):
        if not ticket_input.strip():
            st.warning("Please enter a support ticket message before classifying.")
            return

        engine.uncertainty_threshold = uncertainty_thresh
        result = engine.predict_ticket(ticket_input)

        if not result["success"]:
            st.session_state.single_prediction = None
            st.error(f"Validation Error: {result.get('error')}")
            return
        else:
            st.session_state.single_prediction = result
            st.session_state.single_ticket_text = ticket_input

    if st.session_state.single_prediction is not None:
        result = st.session_state.single_prediction
        active_text = st.session_state.single_ticket_text

        st.markdown("---")
        st.markdown("### 📊 Classification Output")

        # Top Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.metric(label="Predicted Category", value=result["predicted_category"])

        with m_col2:
            st.metric(label="Model Confidence", value=f"{result['confidence']:.1%}")

        with m_col3:
            status_text = "⚠️ Review Required" if result["is_uncertain"] else "✅ Autonomous"
            st.metric(label="Triage Status", value=status_text)

        with m_col4:
            st.metric(label="Priority (Heuristic)", value=result["priority"])

        # Uncertainty alert
        if result["is_uncertain"]:
            st.warning(
                f"**Human Review Flagged:** {result['uncertainty_reason']}  \n"
                "This prediction does not meet the autonomous confidence threshold and requires verification."
            )
        else:
            st.success("Autonomous Prediction: Confidence exceeds safety threshold.")

        # Summary and Priority Policy
        col_sum, col_pol = st.columns(2)
        with col_sum:
            st.info(f"**Issue Summary:** {result['summary']}")
        with col_pol:
            st.caption(f"🛡️ **Priority Policy Notice:** {result['priority_policy_message']}")

        # Class Probabilities Breakdown
        st.markdown("#### Probability Distribution")
        probs_df = pd.DataFrame(
            list(result["all_probabilities"].items()),
            columns=["Category", "Probability"],
        ).sort_values(by="Probability", ascending=False)
        probs_df["Probability (%)"] = (probs_df["Probability"] * 100).round(2)

        st.bar_chart(probs_df.set_index("Category")["Probability (%)"])

        # Manual Correction Panel
        st.markdown("---")
        st.markdown("### ✍️ Human-in-the-Loop Correction Panel")
        st.markdown(
            "If the model prediction is incorrect or uncertain, select the true category to log an auditable correction. "
            "*Original datasets remain immutable.*"
        )

        with st.form(key="correction_form"):
            c_col1, c_col2 = st.columns(2)
            with c_col1:
                corrected_cat = st.selectbox(
                    "Correct Category:",
                    options=APPROVED_CATEGORIES,
                    index=APPROVED_CATEGORIES.index(result["predicted_category"]),
                )
                reviewer_id = st.text_input("Reviewer ID / Agent Name:", value="support_agent")
            with c_col2:
                notes = st.text_input(
                    "Reviewer Notes / Justification:",
                    placeholder="e.g., Customer explicitly mentions unauthorized credit card debit.",
                )
            submit_correction = st.form_submit_button("Submit Correction", type="secondary")

            if submit_correction:
                success = log_manual_correction(
                    ticket_text=active_text,
                    predicted_category=result["predicted_category"],
                    predicted_confidence=result["confidence"],
                    corrected_category=corrected_cat,
                    corrected_by=reviewer_id,
                    review_notes=notes,
                )
                if success:
                    st.success(
                        f"Correction successfully logged! "
                        f"Saved override '{corrected_cat}' to `{RUNTIME_CORRECTIONS_PATH.name}`."
                    )


def render_batch_tab(engine: InferenceEngine):
    """Render batch CSV upload and classification interface."""
    st.subheader("Batch CSV Ticket Classification")
    st.markdown(
        "Upload a CSV file containing incoming customer messages. "
        "The file must contain a `ticket_text`, `text`, or `message` column."
    )

    if "batch_enriched_df" not in st.session_state:
        st.session_state.batch_enriched_df = None
    if "batch_stats" not in st.session_state:
        st.session_state.batch_stats = None

    uploaded_file = st.file_uploader("Upload Ticket CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
            st.write(f"Uploaded file contains **{len(df)}** rows and **{len(df.columns)}** columns.")
            st.dataframe(df.head(3), use_container_width=True)

            if st.button("Process Batch Predictions", type="primary"):
                with st.spinner("Classifying tickets and computing uncertainty metrics..."):
                    success, msg, enriched_df, stats = engine.predict_batch(df)

                if not success:
                    st.error(f"Batch Processing Failed: {msg}")
                    st.session_state.batch_enriched_df = None
                    st.session_state.batch_stats = None
                    return

                st.session_state.batch_enriched_df = enriched_df
                st.session_state.batch_stats = stats
                st.success(msg)

            if st.session_state.batch_enriched_df is not None:
                enriched_df = st.session_state.batch_enriched_df
                stats = st.session_state.batch_stats

                # Flagged / rejected rows expander if any invalid rows existed
                if stats.get("rejected_count", 0) > 0:
                    with st.expander(f"⚠️ {stats['rejected_count']} Invalid Rows Were Flagged and Excluded", expanded=True):
                        st.warning("The following rows contained missing text, stringified nulls, or were below the minimum 5 characters:")
                        st.dataframe(stats["rejected_df"], use_container_width=True)
                        rej_csv = stats["rejected_df"].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download Rejected Rows Log (CSV)",
                            data=rej_csv,
                            file_name="ticketsense_rejected_rows.csv",
                            mime="text/csv",
                        )

                # Batch summary KPIs
                kpi1, kpi2, kpi3 = st.columns(3)
                kpi1.metric("Total Processed", stats["total_processed"])
                kpi2.metric(
                    "Flagged for Human Review",
                    f"{stats['uncertain_count']} ({stats['uncertain_percentage']}%)",
                )
                kpi3.metric("Autonomous Acceptance", f"{100 - stats['uncertain_percentage']:.1f}%")

                st.markdown("#### Category Distribution")
                cat_df = pd.DataFrame(
                    list(stats["category_distribution"].items()),
                    columns=["Category", "Count"],
                ).sort_values(by="Count", ascending=False)
                st.bar_chart(cat_df.set_index("Category")["Count"])

                # Filter options
                st.markdown("#### Enriched Results Table")
                filter_choice = st.radio(
                    "Filter Results:",
                    options=["Show All", "Flagged for Human Review Only", "Autonomous Only"],
                    horizontal=True,
                )

                display_df = enriched_df.copy()
                if filter_choice == "Flagged for Human Review Only":
                    display_df = display_df[display_df["is_uncertain"] == True]
                elif filter_choice == "Autonomous Only":
                    display_df = display_df[display_df["is_uncertain"] == False]

                st.dataframe(display_df, use_container_width=True)

                # Export CSV
                csv_bytes = enriched_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Enriched Predictions (CSV)",
                    data=csv_bytes,
                    file_name="ticketsense_batch_predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error parsing uploaded CSV: {str(e)}")


def render_performance_tab():
    """Render model performance and governance documentation tab."""
    st.subheader("Model Evaluation & Governance")

    # Metrics Overview
    eval_file = REPORTS_DIR / "CATEGORY_MODEL_EVALUATION.md"
    if eval_file.exists():
        with open(eval_file, "r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    else:
        st.info("Evaluation report will appear here once model training is completed.")

    # Confusion matrix visual
    cm_path = REPORTS_DIR / "category_confusion_matrix.png"
    if cm_path.exists():
        st.markdown("### Confusion Matrix (Validation Set)")
        st.image(str(cm_path), caption="Validation Split Confusion Matrix (100 samples)")

    # Governance Notice
    st.markdown("---")
    st.markdown("### 📋 Governance & Gate G1 Requirements")
    st.markdown(
        """
        - **Labeling Policy Review:** The priority classification task is subject to an active review queue (`data/processed/dataset_review_required.csv`).
        - **Class Imbalance:** High-priority records constitute only 6.5% of historical labels, presenting severe risk of false negatives on critical security issues.
        - **Supervisor Sign-off:** Supervised training of the priority classifier remains strictly paused pending final policy determination and sign-off by Technical Supervisor Abdullah Ramzan.
        """
    )


def render_corrections_tab():
    """Render auditable manual corrections log."""
    st.subheader("Auditable Corrections Log")
    st.markdown(
        "Historical record of human agent overrides and feedback. "
        "These records can be utilized for active learning and future retraining cycles without modifying baseline datasets."
    )

    corrections_df = load_corrections_log()

    if corrections_df.empty:
        st.info("No manual corrections have been logged yet. Submit a correction from the Single Ticket tab to see it here.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Corrections Logged", len(corrections_df))
        with col2:
            st.metric("Unique Reviewers", corrections_df["corrected_by"].nunique())

        st.dataframe(corrections_df, use_container_width=True)

        csv_data = corrections_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Corrections Log (CSV)",
            data=csv_data,
            file_name="runtime_corrections.csv",
            mime="text/csv",
        )


def main():
    render_header()
    engine = get_engine()

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Single Ticket",
        "📂 Batch Processing",
        "📈 Performance & Governance",
        "📝 Corrections Log",
    ])

    with tab1:
        render_single_ticket_tab(engine)

    with tab2:
        render_batch_tab(engine)

    with tab3:
        render_performance_tab()

    with tab4:
        render_corrections_tab()


if __name__ == "__main__":
    main()
