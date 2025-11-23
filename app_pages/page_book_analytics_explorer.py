""" Book Analytics Explorer Page Body """
import streamlit as st


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
        "correlate with higher engagement. "
        "Explore correlations, distributions, and trends in "
        "the book catalog data.\n\n"
        "**Success Indicator**:"
        "Correlation ≥ 0.4 between features and engagement."
    )

    st.write("---")
