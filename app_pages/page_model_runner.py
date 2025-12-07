""" Model Runner Page: Predict Engagement for a Selected Book """
import tempfile  # standard import first
import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.exceptions import NotFittedError
from src.analysis.analysis import (
    get_top_predicted_books,
    calculate_genre_entropy
)

HF_BASE_URL = "https://huggingface.co/datasets/revolucia/"
DATASET_PATH = "bookwise-analytics-ml/resolve/main/modeling_data/"
MODEL_PATH = "popularity-score-model/"


def _safe_hf_csv(
    filename: str,
    caption: str = "",
    subfolder: str = "modeling_data"
):
    """
    Load CSV directly from Hugging Face Hub from 'modeling_data' subfolder.
    """
    csv_url = (
        f"{HF_BASE_URL}bookwise-analytics-ml/resolve/main/"
        f"{subfolder}/{filename}"
    )
    try:
        df = pd.read_csv(csv_url)
        if caption:
            st.caption(caption)
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        st.error(
            f"Could not load: {filename} "
            f"from Hugging Face Hub ({subfolder})."
        )
        return None


def _load_hf_model_pkl(filename: str, subfolder: str = "modeling_data"):
    """
    Download and load a pickle model from Hugging Face Hub (model repo).
    """
    # Use the model repo URL, not datasets
    model_url = (
        f"https://huggingface.co/"
        "revolucia/popularity-score-model/resolve/main/"
        f"{subfolder}/{filename}"
    )
    try:
        with requests.get(model_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pkl"
            ) as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name
        model = joblib.load(tmp_path)
        return model
    except (
        requests.RequestException,
        joblib.externals.loky.process_executor.TerminatedWorkerError
    ) as exc:
        st.error(f"Could not load model from Hugging Face: {exc}")
        return None


def page_model_runner_body():
    """
    Run the trained model to help select new titles for the internal catalog.
    """
    st.write("## Catalog Title Selector")
    model_filename = "et_model.pkl"

    st.write(
        "Load model and data from Hugging Face Hub "
        "to support catalog selection."
    )
    # Load data from Hugging Face Hub
    catalog_df = _safe_hf_csv(
        "supply_catalog_analysis.csv",
        caption="Supply Catalog (Hugging Face)"
    )
    features_df = _safe_hf_csv(
        "supply_catalog_final_features.csv",
        caption="Feature Dataset (Hugging Face)"
    )
    if catalog_df is None or features_df is None:
        st.error("Could not load required datasets from Hugging Face.")
        return

    # Load model from Hugging Face (from modeling_data folder)
    model = _load_hf_model_pkl(model_filename)
    if model is None:
        return

    st.write("## Select from Supply Catalog")
    st.write(
        "Review top predicted books and select promising new titles for the "
        "internal catalog."
    )
    try:
        features_path = (
            f"{HF_BASE_URL}{DATASET_PATH}supply_catalog_final_features.csv"
        )
        catalog_path = (
            f"{HF_BASE_URL}{DATASET_PATH}supply_catalog_analysis.csv"
        )
        top_books = get_top_predicted_books(
            model=model,
            features_path=features_path,
            catalog_path=catalog_path,
            top_n=15
        )
        st.write("### Top 15 Recommended Titles")
        st.dataframe(top_books)

        genre_entropy = calculate_genre_entropy(top_books["genres_clean"])
        st.write(f"**Genre Entropy (Top 15):** {genre_entropy:.3f}")

    except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
        st.error(f"Could not compute top predicted books: {e}")

    st.write("### Predict Engagement for a Selected Title")
    st.write(
        "Choose a title from the entire supply catalog to see its predicted "
        "engagement score and decide if it fits the catalog."
    )
    book_titles = catalog_df["title_clean"].dropna().unique()
    selected_title = st.selectbox("Select a book title", book_titles)

    selected_row = (
        catalog_df[catalog_df["title_clean"] == selected_title].iloc[0]
    )
    selected_id = selected_row["goodreads_id_clean"]

    feature_row = features_df[features_df["goodreads_id_clean"] == selected_id]
    if feature_row.empty:
        st.error("No features found for the selected book.")
        return

    features = feature_row.drop(
        labels=[
            "popularity_score",
            "title_clean",
            "goodreads_id_clean"
        ],
        errors="ignore"
    )
    try:
        predicted_score = model.predict(features)[0]
        st.write(f"**Predicted Engagement Score:** {predicted_score:.2f}")
    except (ValueError, TypeError, AttributeError, NotFittedError) as exc:
        st.error(f"Prediction failed: {exc}")
