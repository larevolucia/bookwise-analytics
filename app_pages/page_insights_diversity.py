""" Insights & Diversity Page Body """
import streamlit as st


def page_insights_diversity_body():
    """
    Insights & Diversity Page
    Show genre distribution and fairness metrics
    Addresses Business Requirement 4: Diversity & Fairness
    """

    st.write("## Insights & Diversity")

    st.info(
        "**Business Requirement 4: Diversity & Fairness**\n\n"
        "This page analyzes genre distribution and fairness "
        "in recommendations to ensure balanced representation across  "
        "the catalog. Maintaining diversity prevents over-concentration "
        "in popular genres and provides members with "
        "varied reading experiences.\n\n"
        "**Success Indicator**: Shannon Entropy ≥ baseline (0.7)"
    )

    st.write("---")
