""" Reconnendation Comparison Page Body """
import tempfile
import streamlit as st
import pandas as pd
import joblib
import requests
from src.analysis.analysis import (
    get_top_external_books,
    get_top_predicted_books,
    get_book_predicted_score,
    simulate_uplift,
    calculate_genre_entropy
)


HF_BASE_URL = "https://huggingface.co/datasets/revolucia/"
DATASET_PATH = "bookwise-analytics-ml/resolve/main/modeling_data/"
MODEL_PATH = "popularity-score-model/"


def _safe_hf_csv(filename: str, subfolder: str = "modeling_data"):
    csv_url = (
        f"{HF_BASE_URL}bookwise-analytics-ml/resolve/main/"
        f"{subfolder}/{filename}")
    return pd.read_csv(csv_url)


def _load_hf_model_pkl(filename: str, subfolder: str = "modeling_data"):
    model_url = (
        f"https://huggingface.co/"
        "revolucia/popularity-score-model/resolve/main/"
        f"{subfolder}/{filename}"
    )
    with requests.get(model_url, stream=True, timeout=30) as r:
        r.raise_for_status()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
            for chunk in r.iter_content(chunk_size=8192):
                tmp.write(chunk)
            tmp_path = tmp.name
    return joblib.load(tmp_path)


def page_recommendation_comparison_body():
    """
    Recommendation Comparison Page
    Compare different recommendation strategies
    Addresses Business Requirements 2 & 3: Prediction and Uplift Estimation
    """

    st.write("## Recommendation Comparison")

    st.info(
        "**Business Requirements 2 & 3: "
        "High-Engagement Prediction & Uplift Estimation**\n\n"
        "This page compares three recommendation strategies:\n"
        "* **Editorial Selection**: Books chosen by human curators "
        "(simulated by popularity)\n"
        "* **Predictive Model**: Machine learning-driven recommendations\n\n"
        "**Success Indicators**:\n"
        "* BR-3: Simulated uplift ≥ 10%\n"
        "* BR-4: Genre Entropy is equal to or greater than editorial"
    )

    st.write("---")

    # Load model and datasets
    model_filename = "et_model.pkl"
    features_path = (
        f"{HF_BASE_URL}{DATASET_PATH}supply_catalog_final_features.csv"
        )
    catalog_path = f"{HF_BASE_URL}{DATASET_PATH}supply_catalog_analysis.csv"
    model = _load_hf_model_pkl(model_filename)
    if model is None:
        st.error("Could not load model.")
        return

    # Load features for prediction
    features_df = _safe_hf_csv("supply_catalog_final_features.csv")

    # Get top books recommendations
    top_books = get_top_external_books(catalog_path, top_n=5)
    # Add predicted score for each top book using goodreads_id_clean
    scores = []
    book_id = "goodreads_id_clean"
    for idx, row in top_books.iterrows():
        if book_id in row and book_id in features_df.columns:
            goodreads_id = row["goodreads_id_clean"]
            score = get_book_predicted_score(model, features_df, goodreads_id)
        else:
            score = None
        scores.append(score)
    top_books["predicted_score"] = scores

    # Get model recommendations
    model_books = get_top_predicted_books(
        model=model,
        features_path=features_path,
        catalog_path=catalog_path,
        top_n=5
    )

    # Display comparison table
    st.write("### Recommendation Comparison Table")
    st.write("**Editorial Selection:**")
    st.dataframe(top_books)

    st.write("**Model Recommendations:**")
    st.dataframe(
        model_books[[
            "title_clean",
            "primary_author",
            "genres_clean",
            "predicted_score"
        ]]
    )

    # Simulate uplift
    uplift = simulate_uplift(top_books, model_books)

    # Calculate and display genre entropy
    # For editorial selection
    editorial_entropy = calculate_genre_entropy(top_books["genres_clean"])

    # For model recommendations
    model_entropy = calculate_genre_entropy(model_books["genres_clean"])

    # KPI cards for Uplift and Entropy
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Simulated Uplift (%)",
            value=f"{uplift:.2f}%" if uplift is not None else "N/A",
            help="Percentage increase in predicted engagement from model "
                 "vs editorial"
        )

    with col2:
        st.metric(
            label="Genre Entropy",
            value=f"{model_entropy:.2f}",
            delta=f"{model_entropy - editorial_entropy:+.2f} vs editorial",
            help="Shannon Entropy - Measures diversity. Higher is better."
        )

    st.write("---")
