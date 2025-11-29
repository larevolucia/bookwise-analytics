""" Book Analytics Explorer Page Body """
from pathlib import Path  # stdlib first (pylint C0411)
import pandas as pd
import streamlit as st


def _safe_image(path: Path, caption: str = ""):
    if path.exists():
        st.image(str(path), caption=caption)
    else:
        # fallback for plots that may have been saved in notebook root
        alt = (
            Path("rating_distribution_comparison.png")
            if path.name == "rating_distribution_comparison.png"
            else None
        )
        if alt and alt.exists():
            st.image(str(alt), caption=caption)
        else:
            st.warning(f"Missing plot: {path}")


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

    plots_dir = Path("outputs/eda_plots")

    with st.expander(
        "Catalog Comparisons (genres, authors, publishers, ratings, years)",
        expanded=True,
    ):
        _safe_image(
            plots_dir / "genre_comparison.png",
            caption="Genre distribution: Internal vs Supply",
        )
        _safe_image(
            plots_dir / "author_comparison.png",
            caption="Top authors presence comparison",
        )
        _safe_image(
            plots_dir / "publisher_comparison.png",
            caption="Major vs Indie publisher comparison",
        )
        _safe_image(
            plots_dir / "rating_distribution_comparison.png",
            caption="Rating distribution comparison",
        )
        _safe_image(
            plots_dir / "publication_year_distribution_comparison.png",
            caption="Publication year distribution comparison",
        )
        st.caption(
            "Takeaway: Ratings cluster near 4.0; internal catalog skews "
            "toward mainstream fantasy/fiction; supply offers broader "
            "non-fiction and indie presence. (BR-1)"
        )

    with st.expander("Sample-Size Effects (diagnostics)", expanded=False):
        _safe_image(
            plots_dir / "genre_sample_size_effect.png",
            caption="Genre rating vs volume (Internal)",
        )
        _safe_image(
            plots_dir / "publisher_sample_size_effect.png",
            caption="Publisher rating vs volume (Internal)",
        )
        _safe_image(
            plots_dir / "book_sample_size_effect.png",
            caption="Book rating vs number of ratings (Internal)",
        )
        _safe_image(
            plots_dir / "supply_genre_sample_size_effect.png",
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
            plots_dir / "user_activity_distribution_raw.png",
            caption="User activity distribution (raw)",
        )
        _safe_image(
            plots_dir / "user_activity_distribution_log.png",
            caption="User activity distribution (log)",
        )
        _safe_image(
            plots_dir / "user_genre_preference_analysis.png",
            caption="User genre preference & diversity dashboard",
        )
        _safe_image(
            plots_dir / "genre_engagement_vs_volume_scatter.png",
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
