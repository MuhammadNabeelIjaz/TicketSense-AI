"""Streamlit Interactive Application for TicketSense-AI.

Provides real-time ticket classification, batch CSV processing,
uncertainty routing, manual correction logging, and model governance dashboards.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from ticketsense.config import (
    CATEGORY_MODEL_PATH,
    APPROVED_CATEGORIES,
    APPROVED_PRIORITIES,
    DEFAULT_UNCERTAINTY_THRESHOLD,
    DEFAULT_UNCERTAINTY_MARGIN,
    PRIORITY_POLICY_STATUS_MESSAGE,
    RUNTIME_CORRECTIONS_PATH,
    CATEGORY_METADATA_PATH,
    REPORTS_DIR,
)
from ticketsense.inference.engine import InferenceEngine
from ticketsense.data.corrections import (
    log_manual_correction,
    load_corrections_log,
)
from ticketsense.visualization import (
    plot_class_distribution,
    plot_confusion_matrix_heatmap,
    plot_category_performance,
    plot_experiment_comparison,
    plot_confidence_distribution,
    plot_automatic_vs_manual_review,
)


# Page configuration
st.set_page_config(
    page_title="TicketSense-AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_engine() -> InferenceEngine:
    """Cache inference engine instance across sessions."""
    return InferenceEngine()


def render_header():
    """Render top banner and governance status."""
    st.title("TicketSense-AI")
    st.markdown(
        "**Smart Customer Support Ticket Classifier** | "
        "*Production-grade text classification, uncertainty quantification, and human-in-the-loop review.*"
    )

    meta = {}
    if CATEGORY_METADATA_PATH.exists():
        try:
            with open(CATEGORY_METADATA_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            pass

    arch_name = meta.get("model_name", "Calibrated (Sigmoid 5-Fold CV) TF-IDF + Multinomial Naive Bayes")
    if "MULTINOMIAL_NB" in arch_name:
        arch_name = "Calibrated (Sigmoid 5-Fold CV) TF-IDF + Multinomial Naive Bayes"
    train_samples = meta.get("train_samples", 600)
    val_samples = meta.get("val_samples", 100)
    val_metrics = meta.get("validation_metrics", {})
    val_acc = val_metrics.get("accuracy", 0.93)
    val_f1 = val_metrics.get("macro_f1", 0.9297)

    with st.expander("System Architecture & Governance Status", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                "**Category Classifier:** Active  \n"
                f"- Architecture: {arch_name}  \n"
                f"- Validation Accuracy: **{val_acc*100:.1f}%** | Macro F1: **{val_f1:.3f}**  \n"
                f"- Split: {train_samples} Train, {val_samples} Validation (100 Test isolated)"
            )
        with col2:
            st.markdown(
                "**Priority Classifier:** Active (Two-Stage Rubric ML Model)  \n"
                "- Architecture: Stage 1 (Impact LR) + Stage 2 (Urgency LR) -> Stage 3 (3x3 Matrix)  \n"
                "- Validation Accuracy: **87.0%** | Macro F1: **0.727** | High Precision: **100%**  \n"
                "- Status: *Trained on development rubric labels (Impact x Urgency matrix). Heuristic preserved as fallback.*"
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
        "Load Example Support Ticket:",
        options=list(samples.keys()),
        index=0,
    )

    default_text = samples[selected_sample] if selected_sample != "Custom Message" else ""

    ticket_input = st.text_area(
        "Customer Support Ticket Text:",
        value=default_text,
        height=130,
        placeholder="Type or paste customer inquiry here (minimum 5 characters)...",
    )
    predict_button = st.button("Classify Ticket", type="primary")

    if predict_button:
        if not ticket_input.strip():
            st.error("Please enter a ticket message to classify.")
            st.session_state.single_prediction = None
            st.session_state.single_ticket_text = ""
        else:
            with st.spinner("Analyzing ticket semantics and routing..."):
                result = engine.predict_ticket(ticket_input)
            st.session_state.single_prediction = result
            st.session_state.single_ticket_text = ticket_input

    if st.session_state.single_prediction is not None:
        result = st.session_state.single_prediction
        active_text = st.session_state.single_ticket_text

        st.markdown("---")
        st.markdown("### Classification Output")

        # Top Metric Cards
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)

        with m_col1:
            st.metric(label="Predicted Category", value=result["predicted_category"])

        with m_col2:
            st.metric(label="Model Confidence", value=f"{result['confidence']:.1%}")

        with m_col3:
            status_text = "Review Required" if result["is_uncertain"] else "Autonomous"
            st.metric(label="Triage Status", value=status_text)

        with m_col4:
            st.metric(label="Priority (ML)", value=result["priority"])

        # Intermediate Two-Stage & Priority Triage Metrics
        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
        with p_col1:
            imp_val = result.get("predicted_impact") or "N/A"
            imp_conf = result.get("impact_confidence")
            conf_str = f" ({imp_conf:.1%})" if imp_conf is not None else ""
            st.metric(label="Predicted Impact", value=f"{imp_val}{conf_str}")
        with p_col2:
            urg_val = result.get("predicted_urgency") or "N/A"
            urg_conf = result.get("urgency_confidence")
            conf_str = f" ({urg_conf:.1%})" if urg_conf is not None else ""
            st.metric(label="Predicted Urgency", value=f"{urg_val}{conf_str}")
        with p_col3:
            prio_conf = result.get("priority_confidence")
            prio_conf_str = f"{prio_conf:.1%}" if prio_conf is not None else "N/A"
            st.metric(label="Priority Confidence", value=prio_conf_str)
        with p_col4:
            prio_status_text = "Review Required" if result.get("priority_review_required") else "Autonomous"
            st.metric(label="Priority Triage", value=prio_status_text)

        # Independent Uncertainty Alerts
        cat_flag = result.get("category_review_required", result["is_uncertain"])
        prio_flag = result.get("priority_review_required", False)

        if cat_flag:
            st.warning(
                f"**Category Review Required:** {result['uncertainty_reason']}  \n"
                "Category prediction does not meet autonomous safety threshold and requires agent verification."
            )
        if prio_flag:
            st.warning(
                f"**Priority Review Required:** {result.get('priority_uncertainty_reason', 'Uncertain Priority assignment')}  \n"
                "Priority prediction flagged by safety guardrails and requires supervisor verification."
            )
        if not cat_flag and not prio_flag:
            st.success("Autonomous Acceptance: Both Category and Priority predictions meet all safety thresholds.")

        # Summary and Priority Policy
        col_sum, col_pol = st.columns(2)
        with col_sum:
            st.info(f"**Issue Summary:** {result['summary']}")
        with col_pol:
            st.caption(f"**Priority Policy Notice:** {result['priority_policy_message']}")

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
        st.markdown("### Human-in-the-Loop Correction Panel")
        st.markdown(
            "If the model prediction is incorrect or uncertain, select the true category and/or priority to log an auditable correction. "
            "*Original datasets remain immutable.*"
        )

        with st.form(key="correction_form"):
            c_col1, c_col2, c_col3 = st.columns(3)
            with c_col1:
                corrected_cat = st.selectbox(
                    "Correct Category:",
                    options=APPROVED_CATEGORIES,
                    index=APPROVED_CATEGORIES.index(result["predicted_category"]),
                )
            with c_col2:
                curr_prio = result["priority"]
                prio_idx = APPROVED_PRIORITIES.index(curr_prio) if curr_prio in APPROVED_PRIORITIES else 0
                corrected_prio = st.selectbox(
                    "Correct Priority:",
                    options=APPROVED_PRIORITIES,
                    index=prio_idx,
                )
            with c_col3:
                reviewer_id = st.text_input("Reviewer ID / Agent Name:", value="support_agent")

            notes = st.text_input(
                "Reviewer Notes / Justification:",
                placeholder="e.g., Enterprise outage requires High priority escalation and duplicate billing review.",
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
                    predicted_priority=result["priority"],
                    predicted_priority_confidence=result.get("priority_confidence"),
                    corrected_priority=corrected_prio,
                )
                if success:
                    st.success(
                        f"Correction successfully logged! "
                        f"Saved overrides (Category: '{corrected_cat}', Priority: '{corrected_prio}') to `{RUNTIME_CORRECTIONS_PATH.name}`."
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
                    with st.expander(f"{stats['rejected_count']} Invalid Rows Were Flagged and Excluded", expanded=True):
                        st.warning("The following rows contained missing text, stringified nulls, or were below the minimum 5 characters:")
                        st.dataframe(stats["rejected_df"], use_container_width=True)
                        rej_csv = stats["rejected_df"].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Rejected Rows Log (CSV)",
                            data=rej_csv,
                            file_name="ticketsense_rejected_rows.csv",
                            mime="text/csv",
                        )

                # Batch summary KPIs
                kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
                kpi1.metric("Total Processed", stats["total_processed"])
                kpi2.metric(
                    "Category Review",
                    f"{stats.get('category_review_count', 0)}",
                )
                kpi3.metric(
                    "Priority Review",
                    f"{stats.get('priority_review_count', 0)}",
                )
                kpi4.metric(
                    "Overall Flagged",
                    f"{stats['uncertain_count']} ({stats['uncertain_percentage']}%)",
                )
                kpi5.metric("Fully Autonomous", f"{100 - stats['uncertain_percentage']:.1f}%")

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
                    options=[
                        "Show All",
                        "Flagged for Human Review (Overall)",
                        "Category Review Required",
                        "Priority Review Required",
                        "Autonomous Only",
                    ],
                    horizontal=True,
                )

                display_df = enriched_df.copy()
                if filter_choice == "Flagged for Human Review (Overall)":
                    display_df = display_df[display_df.get("overall_review_required", display_df["is_uncertain"]) == True]
                elif filter_choice == "Category Review Required":
                    display_df = display_df[display_df.get("category_review_required", display_df["is_uncertain"]) == True]
                elif filter_choice == "Priority Review Required":
                    display_df = display_df[display_df.get("priority_review_required", False) == True]
                elif filter_choice == "Autonomous Only":
                    display_df = display_df[display_df.get("overall_review_required", display_df["is_uncertain"]) == False]

                st.caption(
                    "You can directly edit the 'predicted_category' and 'predicted_priority' columns below for any ticket flagged for human review."
                )

                column_config = {
                    "ticket_text": st.column_config.TextColumn("ticket_text", disabled=True),
                    "original_row_number": st.column_config.NumberColumn("original_row_number", disabled=True),
                    "predicted_category": st.column_config.SelectboxColumn(
                        "predicted_category",
                        help="Select corrected category",
                        options=APPROVED_CATEGORIES,
                        required=True,
                    ),
                    "confidence": st.column_config.NumberColumn("confidence", format="%.4f", disabled=True),
                    "predicted_priority": st.column_config.SelectboxColumn(
                        "predicted_priority",
                        help="Select corrected priority",
                        options=APPROVED_PRIORITIES,
                        required=True,
                    ),
                    "priority_confidence": st.column_config.NumberColumn("priority_confidence", format="%.4f", disabled=True),
                    "category_review_required": st.column_config.CheckboxColumn("category_review_required", disabled=True),
                    "priority_review_required": st.column_config.CheckboxColumn("priority_review_required", disabled=True),
                    "overall_review_required": st.column_config.CheckboxColumn("overall_review_required", disabled=True),
                    "uncertainty_reason": st.column_config.TextColumn("uncertainty_reason", disabled=True),
                    "priority_uncertainty_reason": st.column_config.TextColumn("priority_uncertainty_reason", disabled=True),
                }

                disabled_cols = [
                    c for c in display_df.columns
                    if c not in ["predicted_category", "predicted_priority"]
                ]

                edited_display_df = st.data_editor(
                    display_df,
                    column_config=column_config,
                    disabled=disabled_cols,
                    use_container_width=True,
                    key=f"batch_editor_{filter_choice}",
                )

                # Check for human edits in category or priority
                cat_changed = not edited_display_df["predicted_category"].equals(display_df["predicted_category"])
                prio_changed = (
                    "predicted_priority" in edited_display_df.columns
                    and not edited_display_df["predicted_priority"].equals(display_df["predicted_priority"])
                )

                if cat_changed or prio_changed:
                    changed_indices = []
                    for idx in display_df.index:
                        if edited_display_df.loc[idx, "predicted_category"] != display_df.loc[idx, "predicted_category"]:
                            changed_indices.append(idx)
                        elif prio_changed and (edited_display_df.loc[idx, "predicted_priority"] != display_df.loc[idx, "predicted_priority"]):
                            changed_indices.append(idx)
                    changed_indices = list(set(changed_indices))

                    st.info(f"You have modified {len(changed_indices)} ticket prediction(s).")
                    if st.button("Save Human Review Corrections", type="primary", key="save_human_review_btn"):
                        for idx in changed_indices:
                            row = edited_display_df.loc[idx]
                            orig_cat = display_df.loc[idx, "predicted_category"]
                            new_cat = row["predicted_category"]
                            orig_prio = display_df.loc[idx, "predicted_priority"] if "predicted_priority" in display_df.columns else None
                            new_prio = row["predicted_priority"] if "predicted_priority" in row else None
                            ticket_msg = row.get("ticket_text", "")

                            # Update session state master dataframe
                            st.session_state.batch_enriched_df.loc[idx, "predicted_category"] = new_cat
                            if new_prio:
                                st.session_state.batch_enriched_df.loc[idx, "predicted_priority"] = new_prio
                            st.session_state.batch_enriched_df.loc[idx, "is_uncertain"] = False
                            st.session_state.batch_enriched_df.loc[idx, "overall_review_required"] = False
                            st.session_state.batch_enriched_df.loc[idx, "uncertainty_reason"] = "Human Reviewed & Corrected"

                            # Persist audit record
                            log_manual_correction(
                                ticket_text=ticket_msg,
                                predicted_category=orig_cat,
                                predicted_confidence=float(row.get("confidence", 0.0)),
                                corrected_category=new_cat,
                                review_notes=f"Batch UI override for row {row.get('original_row_number', idx)}",
                                predicted_priority=orig_prio,
                                predicted_priority_confidence=float(row.get("priority_confidence", 0.0)) if row.get("priority_confidence") is not None else None,
                                corrected_priority=new_prio,
                            )
                        st.success(f"Successfully saved {len(changed_indices)} human review correction(s) to the audit log!")
                        st.rerun()

                # Export CSV
                csv_bytes = st.session_state.batch_enriched_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Enriched Predictions (CSV)",
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

    st.markdown("---")
    st.markdown("### Production Model Visualizations & Evaluation")

    # Row 1: Dataset Distribution & Model Experiments
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Dataset Class Distribution")
        fig_dist = plot_class_distribution()
        st.pyplot(fig_dist)
        plt.close(fig_dist)
    with col2:
        st.markdown("#### Model Experiment Comparison")
        fig_exp = plot_experiment_comparison()
        st.pyplot(fig_exp)
        plt.close(fig_exp)

    # Row 2: Test Set Confusion Matrix & Category Performance
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Test Set Confusion Matrix Heatmap")
        fig_cm = plot_confusion_matrix_heatmap()
        st.pyplot(fig_cm)
        plt.close(fig_cm)
    with col4:
        st.markdown("#### Category-wise Performance")
        fig_perf = plot_category_performance()
        st.pyplot(fig_perf)
        plt.close(fig_perf)

    # Row 3: Confidence Distribution & Review Split
    col5, col6 = st.columns(2)
    with col5:
        st.markdown("#### Prediction Confidence Distribution")
        fig_conf = plot_confidence_distribution()
        st.pyplot(fig_conf)
        plt.close(fig_conf)
    with col6:
        st.markdown("#### Automatic vs Manual Review Routing")
        fig_rev = plot_automatic_vs_manual_review()
        st.pyplot(fig_rev)
        plt.close(fig_rev)

    # Governance Notice
    st.markdown("---")
    st.markdown("### Governance & Priority Model Policy")
    st.markdown(
        """
        - **Priority ML Model:** Two-Stage Rubric-Based Classifier trained on development split (`models/priority_model.joblib`).
        - **Architecture:** Stage 1 predicts Impact (I1, I2, I3), Stage 2 predicts Urgency (U1, U2, U3), and Stage 3 maps components deterministically through the authoritative 3x3 Rubric matrix.
        - **Labeling Rubric:** Priority labels are rubric-derived (3x3 Impact x Urgency matrix), not operational ground truth.
        - **Class Imbalance Mitigation:** Inverse class weighting (`class_weight='balanced'`) prevents majority class collapse on minority High urgency.
        - **Review Queue:** The 26 candidate review queue records (`data/processed/dataset_review_required.csv`) remain segregated pending supervisor adjudication.
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
            label="Download Corrections Log (CSV)",
            data=csv_data,
            file_name="runtime_corrections.csv",
            mime="text/csv",
        )


def main():
    render_header()
    engine = get_engine()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Single Ticket",
        "Batch Processing",
        "Performance & Governance",
        "Corrections Log",
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
