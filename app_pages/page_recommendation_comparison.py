
""" Reconnendation Comparison Page Body """
import streamlit as st


def page_recommendation_comparison_body():
    """
    Recommendation Comparison Page
    Compare different recommendation strategies
    Addresses Business Requirements 2 & 3: Prediction and Uplift Estimation
    """

    st.write("## Recommendation Comparison")

    st.info(
        "**Business Requirements 2 & 3: "
        "High-Engagement Prediction & Uplift Estimation**\n\n"
        "This page compares three recommendation strategies:\n"
        "* **Random Baseline**: Uniform random selection from the catalog\n"
        "* **Editorial Selection**: Books chosen by human curators "
        "(simulated by popularity)\n"
        "* **Predictive Model**: Machine learning-driven recommendations\n\n"
        "**Success Indicators**:\n"
        "* BR-2: Model RMSE < 1.0 or R² > 0.7\n"
        "* BR-3: Simulated uplift ≥ 10%"
    )

    st.write("---")
