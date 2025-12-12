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
        "**Business Requirements 3 & 4: "
        "Uplift Estimation & Diversity & Fairness**\n\n"
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

    # Select 3 editorial books
    editorial_selected = top_books.head(3)

    # Get top 15 model recommendations
    top_15_model = get_top_predicted_books(
        model=model,
        features_path=features_path,
        catalog_path=catalog_path,
        top_n=15
    )

    # Specify goodreads_id_clean for Dune and Where the Crawdads Sing
    dune_id = "44767458"
    crawdads_id = "36809135"

    # Select these two from the model's top 15
    handpicked_books = top_15_model[
        top_15_model["goodreads_id_clean"]
        .astype(str)
        .isin([dune_id, crawdads_id])
    ]

    # Combine for final editorial list
    final_editorial_list = pd.concat(
        [editorial_selected, handpicked_books],
        ignore_index=True
    )

    # Add predicted score for each book
    scores = []
    for idx, row in final_editorial_list.iterrows():
        if (
            "goodreads_id_clean" in row
            and "goodreads_id_clean" in features_df.columns
        ):
            goodreads_id = row["goodreads_id_clean"]
            score = get_book_predicted_score(model, features_df, goodreads_id)
        else:
            score = None
        scores.append(score)
    final_editorial_list["predicted_score"] = scores

    # Display comparison table
    st.write("### Recommendation Comparison Table")
    st.write("**Editorial Selection:**")
    st.dataframe(final_editorial_list)

    st.write("**Model Recommendations:**")
    st.dataframe(
        model_books[[
            "title_clean",
            "primary_author",
            "genres_clean",
            "predicted_score"
        ]]
    )

    st.write("---")
    # Simulate uplift
    uplift = simulate_uplift(final_editorial_list, model_books)

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

    # Explain how editorial selection is simulated
    st.markdown(
        """
        **How is Editorial Selection Simulated?**

        The editorial selection is designed to mimic how a human editor might
        curate a recommended list. We start by selecting the top 3 books from
        the supply catalog that meet strict editorial criteria, such as recent
        publication, major publisher, award-winning status, and high author
        reputation, sorted by popularity and critical acclaim. To reflect real
        editorial practice, we then supplement this list with 2 handpicked
        titles from the model's top recommendations that are culturally
        relevant or have renewed interest (for example, classics like *Dune*
        due to recent adaptations). This hybrid approach ensures the simulated
        editorial list is both timely and contextually relevant, closely
        matching how editors blend current hits with enduring favorites.
        """
    )

    # Explain how simulate_uplift() works
    st.markdown(
        """
        **How is Simulated Uplift Calculated?**

        The simulated uplift metric shows the percentage increase in the
        average predicted engagement score when switching from editorially
        selected books to those recommended by the predictive model.
        It is computed by comparing the mean predicted scores for both groups,
        helping you assess whether the model's recommendations are expected to
        outperform traditional selections.

        **How is Genre Entropy Calculated?**

        Genre entropy measures the diversity of genres in a set of books using
        Shannon entropy. A higher entropy value indicates a more diverse and
        balanced distribution of genres among the recommended books.

        **Note on Genre Entropy:**

        The genre entropy values shown above for the editorial and model
        selections are lower than the overall catalog entropy reported in
        the EDA notebook. This is expected: entropy measures diversity across
        all genres present, so when calculated on the full catalog (thousands
        of books), the value is higher due to broader genre coverage.
        For reference:

        - **Internal Catalog Genre Entropy:** 4.745
        - **Supply Catalog Genre Entropy:** 4.930

        When comparing only a handful of recommended titles (e.g. 5 books),
        entropy will naturally be lower, as the sample covers fewer genres.

        In this context, a reduction from ~4.7/4.9 (catalog) to values around
        2.32 for the recommendations still indicates a notably diverse
        selection, given the much smaller sample size. Use these values to
        compare diversity **between** recommendation strategies, not against
        the catalog-wide baseline.
        """
    )

    st.write("---")
