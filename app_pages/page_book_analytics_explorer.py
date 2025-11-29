""" Book Analytics Explorer Page Body """
from pathlib import Path  # stdlib first (pylint C0411)
import pandas as pd
import streamlit as st


def _safe_image(filename: str, caption: str = ""):
    # Use Hugging Face URL for images (.webp format)
    hf_base_url = " https://huggingface.co/datasets/revolucia/"
    images_url = f"{hf_base_url}bookwise-analytics-ml/resolve/main/eda_plots/"
    st.image(f"{images_url}{filename}", caption=caption)


def _safe_csv(path: Path, caption: str = "") -> None:
    if path.exists():
        df = pd.read_csv(path)
        st.dataframe(df, use_container_width=True)
        if caption:
            st.caption(caption)
    else:
        st.warning(f"Missing dataset: {path}")


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
        " correlate with higher engagement.\n"
        "Explore correlations, distributions, and trends in "
        "the book catalog data.\n\n"
        "**Success Indicator**: "
        "Correlation ≥ 0.4 between features and engagement."
    )

    st.write("---")

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

    # # will be replaced by modeling artifacts later
    # col1, col2 = st.columns(2)

    # with col1:
    #     # Placeholder: sample of the model table after feature engineering
    #     _safe_csv(
    #         Path("outputs/datasets/modeling/model_features_sample.csv"),
    #         caption="Model features sample (post feature engineering)",
    #     )

    # with col2:
    #     # Placeholder: user behavior cluster dataset
    #     _safe_csv(
    #         Path("outputs/datasets/modeling/user_behavior_clusters.csv"),
    #         caption="User behavior clusters (segmentation output)",
    #     )

    st.info(
        "Next: Surface Bayesian/weighted ratings, "
        "correlations/PPS, and model KPIs. "
        "Tables above will show FE sample and "
        "user cluster outputs once generated."
    )
