""" Book Analytics Explorer Page Body """
import os
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


def _safe_image(filename: str, caption: str = ""):
    """
    Legacy helper: load images from Hugging Face dataset repo.
    Kept for backwards compatibility if you still use HF-hosted plots.
    """
    hf_base_url = "https://huggingface.co/datasets/revolucia/"
    images_url = f"{hf_base_url}bookwise-analytics-ml/resolve/main/eda_plots/"
    st.image(f"{images_url}{filename}", caption=caption)


def _safe_hf_csv(
    filename: str,
    caption: str = "",
    subfolder: str = "datasets",
):
    """
    Load CSV directly from Hugging Face Hub from
    either 'corr_matrix' or 'datasets' subfolder.
    subfolder: "corr_matrix" or "datasets"
    """
    hf_base_url = "https://huggingface.co/datasets/revolucia/"
    csv_url = (
        f"{hf_base_url}bookwise-analytics-ml/resolve/main/"
        f"{subfolder}/{filename}"
    )
    try:
        df = pd.read_csv(csv_url)
        st.dataframe(df, use_container_width=True)
        if caption:
            st.caption(caption)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        st.warning(
            f"Could not load: "
            f"{filename} from Hugging Face Hub ({subfolder}).\n{exc}"
        )


def _safe_hf_csv_heatmap(
    filename: str,
    caption: str = "",
    subfolder: str = "corr_matrix",
):
    """
    Load CSV from Hugging Face Hub and display as a colored heatmap table.
    Used here for correlation matrices.
    """

    hf_base_url = "https://huggingface.co/datasets/revolucia/"
    csv_url = (
        f"{hf_base_url}bookwise-analytics-ml/resolve/main/"
        f"{subfolder}/{filename}"
    )
    try:
        df = pd.read_csv(csv_url, index_col=0)
        numeric_df = df.select_dtypes(include=np.number)
        styled = numeric_df.style.background_gradient(
            cmap="coolwarm", axis=None
        ).format("{:.2f}")
        st.write(f"**{caption}**")
        st.dataframe(styled, use_container_width=True)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        st.warning(
            f"Could not load: "
            f"{filename} from Hugging Face Hub ({subfolder}).\n{exc}"
        )


def _feature_importance_chart():
    """
    Render an interactive Altair bar chart for feature importance
    using the CSV exported from the modeling notebook.

    Expected file path:
        outputs/model_plots/et_feature_importance.csv

    Expected columns (case-insensitive):
        - 'feature'
        - 'importance'
    """
    st.write("### 2. Feature Importance")

    fi_path = "outputs/model_plots/et_feature_importance.csv"

    if not os.path.exists(fi_path):
        st.warning(
            "Feature importance file not found at "
            "`outputs/model_plots/et_feature_importance.csv`.\n\n"
            "Please export it from your modeling notebook to enable this "
            "section."
        )
        return

    df_fi = pd.read_csv(fi_path)

    # Robust column detection (handles e.g. 'Feature' vs 'feature')
    cols_lower = {c.lower(): c for c in df_fi.columns}
    if "feature" not in cols_lower or "importance" not in cols_lower:
        st.error(
            "Feature importance CSV must contain 'feature' and 'importance' "
            "columns (found columns: "
            f"{list(df_fi.columns)})"
        )
        return

    feature_col = cols_lower["feature"]
    importance_col = cols_lower["importance"]

    # Remove rows with missing or non-finite importance values
    df_fi = df_fi[np.isfinite(df_fi[importance_col])]

    # if no valid data remains, show a warning and return
    if df_fi.empty:
        st.error(
            "**No valid feature importance data to display.**\n\n"
            "This may be due to missing, invalid, or all-zero values in your"
            " CSV file. "
            "Please check that `outputs/model_plots/et_feature_importance.csv`"
            " exists, has the correct columns, and contains valid numeric "
            "values."
        )
        # show debug info for developers
        st.write("**Debug info:**")
        st.write("CSV columns:", list(cols_lower.values()))
        st.write("First 5 rows of raw data:")
        st.write(pd.read_csv(fi_path).head())
        return

    # Sort in descending order of importance
    df_fi = df_fi.sort_values(by=importance_col, ascending=False)

    # Show only the top 20 features
    df_fi = df_fi.head(20)

    st.markdown(
        "This chart shows which features the ExtraTrees model considered most "
        "important when predicting the engagement metric "
        "(`popularity_score`). "
        "It complements the correlation analysis by capturing nonlinear "
        "effects and interactions."
    )

    chart = (
        alt.Chart(df_fi)
        .mark_bar()
        .encode(
            x=alt.X(
                f"{importance_col}:Q",
                title="Importance score",
            ),
            y=alt.Y(
                f"{feature_col}:N",
                sort="-x",
                title="Feature",
            ),
            tooltip=[feature_col, importance_col],
        )
        .properties(height=500)
    )

    st.altair_chart(chart, use_container_width=True)

    st.info(
        "**Feature Importance Summary**\n"
        "- External behavioral signals dominate the top positions.\n"
        "- Features such as rating volume, votes, and like percentage provide "
        "the strongest contribution to engagement prediction.\n"
        "- Metadata fields (e.g. awards, genre count, publication year) play "
        "a secondary role compared to external popularity signals."
    )


def _shap_summary_section():
    """
    Display a SHAP summary plot image generated in the modeling notebook.

    Expected image path:
        outputs/model_plots/et_shap_summary.png

    You can adjust the filename/extension here to match what you exported.
    """
    st.write("### 3. SHAP Summary Plot")

    shap_path = "outputs/model_plots/et_shap_summary.png"

    if not os.path.exists(shap_path):
        st.warning(
            "SHAP summary plot not found at "
            "`outputs/model_plots/et_shap_summary.png`.\n\n"
            "Export a SHAP summary plot in your modeling notebook and save it "
            "to this path to enable this section."
        )
        return

    st.markdown(
        "SHAP (SHapley Additive exPlanations) provides a model-agnostic way "
        "to understand how each feature influences the prediction. "
        "The summary plot below shows:\n\n"
        "- **Which features** have the largest overall impact.\n"
        "- **Direction of effect** (whether high values tend to increase or "
        "decrease the engagement prediction).\n"
        "- **Distribution of feature impact** across all books."
    )

    st.image(
        shap_path,
        caption="SHAP Summary Plot: ExtraTrees engagement model",
        width='content',
    )

    st.info(
        "Together with the feature importance chart, the SHAP summary "
        "confirms that external popularity signals drive the model's "
        "decisions, while metadata has a more modest but still "
        "interpretable contribution."
    )


def page_book_analytics_explorer_body():
    """
    Book Analytics Explorer Page

    Focus:
        - Business Requirement BR-1:
        Identify features correlated with engagement.
        - Show both correlation analysis and model-based feature importance
          (including SHAP) to explain what drives `popularity_score`.
    """

    st.write("## Book Analytics Explorer")

    st.info(
        "**Business Requirement 1: Engagement Drivers**\n\n"
        "This page identifies which book features are associated with higher "
        "engagement, measured by a composite metric (`popularity_score`). "
        "It combines correlation analysis with model-based feature importance "
        "and SHAP explanations.\n\n"
        "**Success Indicator:** At least one feature should show correlation "
        "≥ 0.4 with `popularity_score`, and the ML model should highlight "
        "consistent drivers of engagement."
    )

    # 1. Correlation Analysis
    st.write("---")
    st.write("### 1. Correlation Analysis")

    st.markdown(
        "The matrices below show how internal Goodreads features, external "
        "signals, and metadata relate to the engagement target "
        "(`popularity_score`).\n\n"
        "- Internal features capture in-platform ratings and behavior.\n"
        "- External features capture broader social proof and popularity.\n"
        "- Metadata describes the book itself (e.g. awards, genres, etc.).\n\n"
        "In the notebooks, `popularity_score` was shown to have much stronger "
        "correlations with external signals than with raw average ratings "
        "alone, motivating its use as the main engagement target."
    )

    _safe_hf_csv_heatmap(
        "internal_corr_matrix.csv",
        caption="Internal Goodreads Features: Correlation Matrix",
    )
    _safe_hf_csv_heatmap(
        "external_corr_matrix.csv",
        caption="External Popularity Features: Correlation Matrix",
    )
    _safe_hf_csv_heatmap(
        "metadata_corr_matrix.csv",
        caption="Metadata Features: Correlation Matrix",
    )

    st.success(
        "**Correlation Conclusion (BR-1):**\n"
        "- External popularity signals (e.g. rating volume, votes, like "
        "percentage) are the **only features** exceeding the 0.4 correlation "
        "threshold with `popularity_score`.\n"
        "- Metadata and internal-only fields show weaker associations and "
        "cannot explain engagement on their own.\n"
        "- This confirms that the business should rely on external signals to "
        "identify high-engagement titles."
    )

    # 2. Feature Importance (Model-Based)
    st.write("---")
    _feature_importance_chart()

    # 3. SHAP Summary Plot
    st.write("---")
    _shap_summary_section()

    # 4. Overall BR-1 Summary

    st.write("---")
    st.write("### 4. Overall Summary: Engagement Drivers")

    st.success(
        "The combination of **correlation analysis**, **feature importance**, "
        "and **SHAP explanations** provides a consistent answer to BR-1:\n\n"
        "- Engagement is primarily driven by **external popularity and social "
        "proof signals**, not by intrinsic metadata alone.\n"
        "- The ML model successfully captures these relationships and uses "
        "them to predict `popularity_score`, which is later consumed by the "
        "Recommendation and Model Runner pages.\n\n"
        "These insights guide both **acquisition** (which titles to license) "
        "and **editorial curation** (which books to promote to members)."
    )
