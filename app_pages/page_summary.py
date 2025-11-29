""" Page: Executive Summary Body """
import streamlit as st
import pandas as pd


def page_summary_body():
    """
    Executive Summary Page
    Display key project information and KPIs
    """

    st.write("## Executive Summary")

    st.info(
        "**Project Overview:**\n\n"
        "This project simulates a subscription-based book club "
        " where members receive one monthly 'credit' to select a book "
        "from a curated catalog.\n"
        "Despite stable subscriber numbers, "
        "engagement and redemption rates are declining, often due "
        "to poor book-member matches.\n\n"
        "The goal is to transition from intuition-driven curation "
        "to data-driven selection, using predictive analytics "
        "to increase satisfaction, loyalty, and catalog diversity."
    )

    # Project Terms & Jargon
    st.write("---")
    st.write("### Project Terms & Jargon")

    with st.expander("Click to expand project terminology"):
        st.markdown(
            """
            * **Member**: A subscribed user of the book club service.
            * **Uplift**: The improvement in engagement scores when comparing
              algorithmic recommendations to baseline strategies
              (editorial or random)."
            * **Genre Entropy**: A diversity metric measuring how evenly
               distributed recommendations are across genres.
               Higher entropy indicates greater diversity.
            * **Editorial Selection**: Books chosen by human curators based
               on intuition and expertise.
            * **Model-Driven Selection**: Books recommended by
               the predictive analytics algorithm based on
               historical data patterns.
            """
        )

    # Dataset Information
    st.write("---")
    st.write("### Project Dataset")

    st.write(
        "This project integrates multiple publicly available sources "
        "to emulate both internal system activity and catalog-wide "
        "data of a subscription book service:"
    )

    dataset_info = {
        "Dataset": [
            "Best Books Ever (BBE)",
            "Goodbooks-10k",
            "Open Library API",
            "Google Books API"
        ],
        "Source": [
            "GitHub",
            "GitHub",
            "Open Library",
            "Google Books"
        ],
        "Purpose": [
            "Emulate curated book catalog with quality indicators",
            "Simulate user-behavioral data (rating interactions)",
            "Enrich metadata with descriptive information",
            "Enrich metadata with descriptive information"
        ]
    }

    st.table(pd.DataFrame(dataset_info))

    st.write(
        "* The dataset contains information about books including ratings, "
        "reviews, genres, publication dates, and other features.\n"
        "* An 8K overlap subset is used for engagement modeling and "
        "behavioral analysis."
    )

    # Business Requirements
    st.write("---")
    st.write("### Business Requirements")

    st.success(
        "**BR-1: Engagement Drivers**\n\n"
        "Identify which book and genre features correlate"
        " with higher engagement.\n"
        "* **Success Indicator**: Correlation ≥ 0.4 between "
        "features and engagement.\n\n"

        "**BR-2: High-Engagement Prediction**\n\n"
        "Predict which titles are most likely to achieve high engagement "
        "based on historical data.\n"
        "* **Success Indicator**: Model RMSE < 1.0 or R² > 0.7.\n\n"

        "**BR-3: Uplift Estimation**\n\n"
        "Estimate potential retention uplift from algorithmic vs "
        "manual (editorial) selection.\n"
        "* **Success Indicator**: Simulated uplift ≥ 10%.\n\n"

        "**BR-4: Diversity & Fairness**\n\n"
        "Maintain diversity and fairness in recommendations across genres.\n"
        "* **Success Indicator**: Shannon Entropy ≥ baseline (0.7)."
    )

    # Key Performance Indicators
    st.write("---")
    st.write("### Key Performance Indicators (KPIs)")

    # Create three columns for KPI cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Model RMSE",
            value="0.85",  # Replace with actual value from model
            delta="-0.15 vs baseline",
            help="Root Mean Square Error (Low = better) Target: < 1.0"
        )

    with col2:
        st.metric(
            label="Model R² Score",
            value="0.73",  # Replace with actual value from model
            delta="+0.23 vs baseline",
            help="Coefficient of Determination (High = better) Target: > 0.7"
        )

    with col3:
        st.metric(
            label="Genre Entropy",
            value="0.78",  # Replace with actual value
            delta="+0.08 vs baseline",
            help="Shannon Entropy - Measures diversity. Target: ≥ 0.7"
        )

    # Model Performance Comparison
    st.write("---")
    st.write("### Model Performance vs Baseline")

    st.write(
        "The chart below compares the average predicted engagement  "
        "scores across different recommendation strategies:"
    )

    # Create sample data for comparison chart

    # Additional Information
    st.write("---")
    st.write("### Project Hypotheses")

    with st.expander("View tested hypotheses and validation methods"):
        st.markdown(
            """
            **H1: Multi-Genre & High-Rated Books Drive Engagement**
            * **Hypothesis**: Books with high cross-platform ratings (>4.0) and
              multi-genre tags achieve higher engagement.
            * **Validation**: Correlation and multiple regression analysis.
            * **Expected Outcome**: Positive correlation (r > 0.4).

            **H2: Historical Patterns Predict Engagement**
            * **Hypothesis**: Historical rating and review patterns can predict
              engagement with ~80% accuracy.
            * **Validation**: Regression models (XGBoost, Random Forest).
            * **Expected Outcome**: Model achieves RMSE < 1.0 or R² > 0.7.

            **H3: Recent Publications Yield Higher Satisfaction**
            * **Hypothesis**: Recent publications yield higher satisfaction.
            * **Validation**: Correlation and time-series analysis.
            * **Expected Outcome**: Negative correlation between publication
               date and satisfaction.

            **H4: Algorithmic Selection Increases Engagement**
            * **Hypothesis**: Algorithmic selection based on predicted
              engagement increases overall engagement by at least 10%
              compared to editorial or random selection.
            * **Validation**: Simulated uplift modeling using
               engagement-retention proxy.
            * **Expected Outcome**: ≥10% (uplift in simulated engagement).
            """
        )

    # Navigation Hints
    st.write("---")
    st.info(
        "**Next Steps**: Use the navigation menu on the left to:\n"
        "* Explore features and correlations in **Book Analytics Explorer**\n"
        "* Compare recommendation types in **Recommendation Comparison**\n"
        "* Analyze genre diversity and fairness in **Insights & Diversity**"
    )
