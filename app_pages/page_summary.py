""" Page: Executive Summary Body """
import os
import json
import streamlit as st
import pandas as pd


def page_summary_body():
    """
    Executive Summary Page
    Display key project information and KPIs
    """

    # --- Load model metrics ---
    model_metrics_path = os.path.join(
        "outputs",
        "models",
        "metrics.json"
    )
    baseline_metrics_path = os.path.join(
        "outputs",
        "models",
        "baseline_metrics.json"
    )

    # defaults
    model_rmse = model_r2 = model_mae = "N/A"
    baseline_rmse = baseline_r2 = baseline_mae = None

    # load model metrics
    if os.path.exists(model_metrics_path):
        with open(model_metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        model_rmse = round(metrics.get("rmse", 0), 2)
        model_r2 = round(metrics.get("r2", 0), 2)
        model_mae = round(metrics.get("mae", 0), 2)

    # Load baseline metrics
    if os.path.exists(baseline_metrics_path):
        with open(baseline_metrics_path, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        baseline_rmse = baseline.get("rmse", None)
        baseline_r2 = baseline.get("r2", None)
        baseline_mae = baseline.get("mae", None)

    # Compute deltas if possible
    if (
        all(x is not None for x in [baseline_rmse, model_rmse])
        and model_rmse != "N/A"
    ):
        delta_rmse = round(model_rmse - baseline_rmse, 2)
        delta_rmse_str = f"{delta_rmse:+.2f} vs baseline"
    else:
        delta_rmse_str = "N/A"

    if (
        all(x is not None for x in [baseline_r2, model_r2])
        and model_r2 != "N/A"
    ):
        delta_r2 = round(model_r2 - baseline_r2, 2)
        delta_r2_str = f"{delta_r2:+.2f} vs baseline"
    else:
        delta_r2_str = "N/A"

    if (
        all(x is not None for x in [baseline_mae, model_mae])
        and model_mae != "N/A"
    ):
        delta_mae = round(model_mae - baseline_mae, 2)
        delta_mae_str = f"{delta_mae:+.2f} vs baseline"
    else:
        delta_mae_str = "N/A"

    st.write("## Executive Summary")

    st.info(
        "**Project Overview:**\n\n"
        "This project simulates a subscription-based book club "
        "where members receive one monthly 'credit' to select a book "
        "from a curated catalog.\n"
        "Despite stable subscriber numbers, engagement and redemption rates "
        "are declining, often due to poor book-member matches.\n\n"
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
              (editorial or random).
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
        "Identify which metadata and external features correlate "
        "with higher engagement.\n"
        "* **Success Indicator**: Correlation ≥ 0.4 between features "
        "and engagement.\n\n"

        "**BR-2: High-Engagement Prediction**\n\n"
        "Predict which titles are most likely to achieve high engagement "
        "based on historical data.\n"
        "* **Success Indicator**: Model RMSE < 1.0 or R² > 0.7.\n\n"

        "**BR-3: Uplift Estimation**\n\n"
        "Estimate potential retention uplift from algorithmic vs manual "
        "(editorial) selection.\n"
        "* **Success Indicator**: Simulated uplift ≥ 10%.\n\n"

        "**BR-4: Diversity & Fairness**\n\n"
        "Maintain diversity and fairness in recommendations across genres.\n"
        "* **Success Indicator**: Shannon Entropy ≥ editorial baseline."
    )

    # Key Performance Indicators (actual values from 06_Modeling.ipynb)
    st.write("---")
    st.write("### Key Performance Indicators (KPIs)")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            label="Model RMSE",
            value=model_rmse,
            delta=delta_rmse_str,
            help="Root Mean Square Error (Low = better) Target: < 1.0",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            label="Model R² Score",
            value=model_r2,
            delta=delta_r2_str,
            help="Coefficient of Determination (High = better) Target: > 0.7"
        )

    with col3:
        st.metric(
            label="Model MAE",
            value=model_mae,
            delta=delta_mae_str,
            help="Mean Absolute Error (Low = better)",
            delta_color="inverse"
        )

    # Model Selection and Correlation Insights
    st.write("---")
    st.write("### Model Selection & Insights")

    st.markdown(
        """
        The best-performing model was **ExtraTreesRegressor**,
        achieving an R² of 0.81 and RMSE of 0.94.
        - **Top predictive features:** External engagement metrics
        (`numratings_log`, `external_popularity_score`,
        `external_bbe_ratings_5`).

        - **Correlation Insights:** Only external signals
        (such as `external_numratings_log`, `external_votes_log`, and
        `external_likedpct`) have correlation coefficients ≥ 0.4 with
        the engagement metric (`popularity_score`). No metadata features
        reach this threshold; the closest is `has_award_final`
        (correlation = 0.34). This highlights the importance of external
        popularity and social proof signals in predicting engagement.

        - **Recency Importance:** External features dominate the top
        predictors, but and publication recency (2010) also plays a
        significant role, indicating that newer books tend to engage
        members more effectively.
        """
    )

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
            * **Validation**: Regression models (Random Forest, Extra Trees, "
            "Gradient Boosting).
            * **Expected Outcome**: Model achieves RMSE < 1.0 or R² > 0.7.

            **H3: Recent Publications Yield Higher Satisfaction**
            * **Hypothesis**: Recent publications yield higher satisfaction.
            * **Validation**: Feature importance analysis of publication
            decade.
            * **Expected Outcome**: Publication recency (e.g., 2010s) appears
            among top features with moderate importance.

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
        "* Analyze book prediction scores in **Model Runner**"
    )
