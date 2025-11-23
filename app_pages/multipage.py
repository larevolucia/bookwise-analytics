""" MultiPage Class for Streamlit Applications"""
import streamlit as st


class MultiPage:
    """
    Class to generate multiple Streamlit pages using
    an object oriented approach
    """

    def __init__(self, app_name) -> None:
        self.pages = []
        self.app_name = app_name

        st.set_page_config(
            page_title=self.app_name,
            page_icon="📚",
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def add_page(self, title, func) -> None:
        """
        Add pages to the project
        """
        self.pages.append({"title": title, "function": func})

    def run(self):
        """
        Run the MultiPage app
        """
        st.title(self.app_name)
        page = st.sidebar.radio(
            'Navigation',
            self.pages,
            format_func=lambda page: page['title']
        )
        page['function']()
