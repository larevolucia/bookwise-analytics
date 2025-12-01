""" Book Analytics Explorer Page Body """
import pandas as pd
import streamlit as st
import numpy as np


def _safe_image(filename: str, caption: str = ""):
    # Use Hugging Face URL for images (.webp format)
    hf_base_url = "https://huggingface.co/datasets/revolucia/"
    images_url = (
        f"{hf_base_url}bookwise-analytics-ml/resolve/main/eda_plots/"
    )
    st.image(f"{images_url}{filename}", caption=caption)


def _safe_hf_csv(
    filename: str,
    caption: str = "",
    subfolder: str = "datasets"
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
    subfolder: str = "corr_matrix"
):
    """
    Load CSV from Hugging Face Hub and display as a colored heatmap table.
    """

    hf_base_url = "https://huggingface.co/datasets/revolucia/"
    csv_url = (
        f"{hf_base_url}bookwise-analytics-ml/resolve/main/"
        f"{subfolder}/{filename}"
    )
    try:
        df = pd.read_csv(csv_url, index_col=0)
        # Only color numeric columns
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


def page_book_analytics_explorer_body():
    """
    Book Analytics Explorer Page
    Explore correlations and trends
    Addresses Business Requirement 1: Engagement Drivers
    """

    st.write("## Book Analytics Explorer")

    st.info(
        "**Business Requirement 1: Engagement Drivers**\n\n"
        "This page helps identify which book and genre features "
        "correlate with higher engagement.\n"
        "Explore correlations, distributions, and trends in "
        "the book catalog data.\n\n"
        "**Success Indicator**: "
        "Correlation ≥ 0.4 between features and engagement."
    )

    st.write("---")
    st.write("Correlation Matrices")
    
    st.markdown(
        "#### Correlation Analysis Summary\n"
        "The correlation matrices above show how internal, external, "
        "and metadata features relate to key engagement signals. "
        "Analysis reveals that `popularity_score` consistently demonstrates "
        "stronger and broader correlations with predictive features "
        "compared to `gb_rating_clean`. While average ratings "
        "(`gb_rating_clean`) are weakly associated with most metadata and "
        "external signals,`popularity_score` captures a more comprehensive "
        "picture of book success, reflecting both user engagement and "
        "visibility. As a composite metric, it is less affected by "
        "sample-size bias and rating inflation, making it a more robust and "
        "informative target for downstream modeling and business decisions."
    )

    _safe_hf_csv_heatmap(
        "internal_corr_matrix.csv",
        caption="Internal GB Features Correlation Matrix"
    )
    _safe_hf_csv_heatmap(
        "external_corr_matrix.csv",
        caption="External Features Correlation Matrix"
    )
    _safe_hf_csv_heatmap(
        "metadata_corr_matrix.csv",
        caption="Metadata Features Correlation Matrix"
    )

    st.write("---")

    st.markdown(
        "#### Catalog Comparison Visualizations\n"
        "The images below compare catalog characteristics such as genres, "
        "authors, publishers, ratings, and publication years. "
        "These visualizations highlight differences between the internal "
        "catalog and the broader supply, showing trends in genre distribution,"
        " author prominence, publisher types, and rating patterns. They "
        "provide context for understanding engagement and diversity "
        "in the dataset."
    )

    with st.expander(
        "Catalog Comparisons (genres, authors, publishers, ratings, years)",
        expanded=True,
    ):
        _safe_image(
            "genre_comparison.webp",
            caption="Genre distribution: Internal vs Supply",
        )
        _safe_image(
            "author_comparison.webp",
            caption="Top authors presence comparison",
        )
        _safe_image(
            "publisher_comparison.webp",
            caption="Major vs Indie publisher comparison",
        )
        _safe_image(
            "rating_distribution_comparison.webp",
            caption="Rating distribution comparison",
        )
        _safe_image(
            "publication_year_distribution_comparison.webp",
            caption="Publication year distribution comparison",
        )
        st.caption(
            "Takeaway: Ratings cluster near 4.0; internal catalog skews "
            "toward mainstream fantasy/fiction; supply offers broader "
            "non-fiction and indie presence. (BR-1)"
        )

    with st.expander("Sample-Size Effects (diagnostics)", expanded=False):
        _safe_image(
            "genre_sample_size_effect.webp",
            caption="Genre rating vs volume (Internal)",
        )
        _safe_image(
            "publisher_sample_size_effect.webp",
            caption="Publisher rating vs volume (Internal)",
        )
        _safe_image(
            "book_sample_size_effect.webp",
            caption="Book rating vs number of ratings (Internal)",
        )
        _safe_image(
            "supply_genre_sample_size_effect.webp",
            caption="Genre rating vs volume (Supply)",
        )
        st.caption(
            "Low-volume entities inflate averages; reliability increases "
            "with volume. Prefer Bayesian/weighted ratings and log "
            "transforms in modeling."
        )

    with st.expander(
        "User Behavior (activity, preferences, diversity)",
        expanded=False,
    ):
        _safe_image(
            "user_activity_distribution_raw.webp",
            caption="User activity distribution (raw)",
        )
        _safe_image(
            "user_activity_distribution_log.webp",
            caption="User activity distribution (log)",
        )
        _safe_image(
            "user_genre_preference_analysis.webp",
            caption="User genre preference & diversity dashboard",
        )
        _safe_image(
            "genre_engagement_vs_volume_scatter.webp",
            caption="Genre engagement vs catalog volume",
        )
        st.caption(
            "Readers show high apparent genre diversity due to multi-tagging, "
            "with engagement concentrated in broad genres (fiction, fantasy, "
            "classics)."
        )

    st.write("---")
    st.write("Data tables")

    st.info(
        "Next: Surface Bayesian/weighted ratings, "
        "correlations/PPS, and model KPIs. "
        "Tables above will show FE sample and "
        "user cluster outputs once generated."
    )
