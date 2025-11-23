""" Main application script for BookWise Analytics Streamlit app """
from app_pages.multipage import MultiPage

# Load pages scripts
from app_pages.page_summary import page_summary_body
from app_pages.page_book_analytics_explorer import (
    page_book_analytics_explorer_body
    )
from app_pages.page_recommendation_comparison import (
    page_recommendation_comparison_body
)
from app_pages.page_insights_diversity import page_insights_diversity_body

# create an instance of the app
app = MultiPage(app_name="BookWise Analytics")

app.add_page("Executive Summary", page_summary_body)
app.add_page("Book Analytics Explorer", page_book_analytics_explorer_body)
app.add_page("Recommendation Comparison", page_recommendation_comparison_body)
app.add_page("Insights & Diversity", page_insights_diversity_body)

app.run()
