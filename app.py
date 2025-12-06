# pylint: disable=unused-import
""" Main application script for BookWise Analytics Streamlit app """
# required for running modeling pipeline
from src.modeling.modeling_pipeline import DropColumns  # noqa: F401
from src.modeling.modeling_pipeline import FillNAWithValue  # noqa: F401
from src.modeling.modeling_pipeline import AddMissingFlags  # noqa: F401

# Load pages scripts
from app_pages.multipage import MultiPage
from app_pages.page_summary import page_summary_body
from app_pages.page_book_analytics_explorer import (
    page_book_analytics_explorer_body
    )
from app_pages.page_recommendation_comparison import (
    page_recommendation_comparison_body
)
from app_pages.page_insights_diversity import page_insights_diversity_body
from app_pages.page_model_runner import page_model_runner_body

# create an instance of the app
app = MultiPage(app_name="BookWise Analytics")

app.add_page("Executive Summary", page_summary_body)
app.add_page("Book Analytics Explorer", page_book_analytics_explorer_body)
app.add_page("Recommendation Comparison", page_recommendation_comparison_body)
app.add_page("Insights & Diversity", page_insights_diversity_body)
app.add_page("Model Runner", page_model_runner_body)

app.run()
