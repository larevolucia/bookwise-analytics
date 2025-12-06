""" Functions for analyzing book datasets and model predictions. """
import pandas as pd
import numpy as np


def calculate_genre_entropy(genres):
    """
    Calculate Shannon entropy for a list or pandas Series of genres.
    """
    _, counts = np.unique(genres, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def get_top_predicted_books(
    model,
    features_path: str,
    catalog_path: str,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Run the model on the entire supply catalog features,
    map predictions to supply_catalog_analysis using goodreads_id_clean,
    and return the top N rated books with selected columns.

    Returns DataFrame with columns:
    ['title_clean', 'primary_author', 'genres_clean', 'predicted_score']
    """

    # Load features and catalog
    features_df = pd.read_csv(features_path)
    catalog_df = pd.read_csv(catalog_path)

    # Filter both DataFrames to is_overlap == False (PEP8-compliant)
    if "is_overlap" in features_df.columns:
        features_df = features_df[~features_df["is_overlap"]]
    if "is_overlap" in catalog_df.columns:
        catalog_df = catalog_df[~catalog_df["is_overlap"]]

    # Prepare features for prediction
    features_for_pred = features_df.drop(
        columns=["popularity_score", "title_clean", "goodreads_id_clean"],
        errors="ignore"
    )
    features_df["predicted_score"] = model.predict(features_for_pred)

    # Merge predictions with catalog on goodreads_id_clean
    merged = pd.merge(
        features_df,
        catalog_df[[
            "goodreads_id_clean",
            "title_clean",
            "primary_author",
            "genres_clean",
        ]],
        on="goodreads_id_clean",
        how="left"
    )

    # Sort by predicted_score and return top N
    top_books = (
        merged.sort_values("predicted_score", ascending=False).head(top_n)
    )
    return top_books[[
        "title_clean",
        "primary_author",
        "genres_clean",
        "predicted_score",
    ]]
