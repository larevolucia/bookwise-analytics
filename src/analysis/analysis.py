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


def get_top_external_books(catalog_path, top_n=5):
    """
    Returns the top N books from the supply catalog where:
      - is_overlap == False
      - has_award_encoded == 1
      - is_top_author == 1
      - is_major_publisher == True
      - publication_decade in [2000, 2010]
      - numRatings_clean > 50
    Sorted by "rating_clean" (descending).
    """
    df = pd.read_csv(catalog_path)
    # Debug: show columns
    print("Catalog columns:", list(df.columns))
    # Check for required columns
    required = [
        "is_overlap", "rating_clean", "has_award_encoded",
        "is_top_author", "is_major_publisher", "publication_decade",
        "numRatings_clean"
    ]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing in catalog: {missing}")

    # Apply editorial filters including numRatings_clean > 50
    filtered = df[
        (~df["is_overlap"]) &
        (df["has_award_encoded"] == 1) &
        (df["is_major_publisher"]) &
        (df["publication_decade"].isin([1990, 2000, 2010])) &
        (df["numRatings_log"] > 10) &
        (df["bbeScore_clean"] > 1000)
    ]
    top_books = filtered.sort_values(
        "rating_clean",
        ascending=False
    ).head(top_n)
    # Return only relevant columns (add more if needed)
    return top_books[
        [
            "goodreads_id_clean",
            "title_clean",
            "primary_author",
            "publication_decade",
            "genres_clean",
        ]
    ]


def get_book_predicted_score(
    model,
    features_df: pd.DataFrame,
    goodreads_id: str
) -> float:
    """
    Get the predicted score for a specific book by goodreads_id_clean.
    Returns None if not found.
    """
    row = features_df[features_df["goodreads_id_clean"] == goodreads_id]
    if row.empty:
        return None
    features = row.drop(
        columns=["popularity_score", "title_clean", "goodreads_id_clean"],
        errors="ignore"
    )
    return float(model.predict(features)[0])


def simulate_uplift(
    editorial_df: pd.DataFrame, model_df: pd.DataFrame
) -> float:
    """
    Simulate uplift as the percentage increase in mean predicted score
    from editorial selection to model recommendations.
    Returns uplift as a float (percentage).
    """
    editorial_mean = editorial_df["predicted_score"].mean()
    model_mean = model_df["predicted_score"].mean()
    if editorial_mean == 0 or pd.isnull(editorial_mean):
        return None
    uplift = (model_mean - editorial_mean) / abs(editorial_mean) * 100
    return uplift
