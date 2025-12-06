""" Feature engineering functions for modeling datasets. """
import numpy as np
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer
    ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except ImportError:
    ST_MODEL = None


def fe_engineering(
    df,
    encode_text_embeddings=False,
    top_n_authors=50,
    top_n_genres=30,
    bool_cols=None,
    text_col='description_clean',
    genres_col='genres_clean',
    author_col='author_clean',
    publisher_col='publisher_clean',
    series_col='series_clean',
):
    """
    Apply DRY feature engineering to any cleaned or modeling dataset.
    All operations are optional, modular, and safe for missing columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    encode_text_embeddings : bool
        Whether to compute text embeddings. Default False.
    top_n_authors : int
        Number of top authors to consider for popularity features. Default 50.
    top_n_genres : int
        Number of top genres to consider for genre features. Default 30.
    bool_cols : list of str
        List of boolean columns to process.
    text_col : str
        Column with clean description text.
    genres_col : str
        Genres list or string column.
    author_col : str
        Author name column.
    publisher_col : str
        Publisher column.
    series_col : str
        Series metadata column.


    Returns
    -------
    df : pd.DataFrame
        Dataset with engineered features added.
    """

    df = df.copy()

    print("Starting feature engineering...")

    # boolean columns
    if bool_cols is None:
        bool_cols = [col for col in df.columns if df[col].dtype == bool]

    if bool_cols:
        print("Computing boolean encodings...")
        for col in bool_cols:
            df[f"{col}_encoded"] = df[col].astype(int)
            print(f"  ✓ Encoded: {col} -> {col}_encoded")
        print("Completed boolean encodings.")

    # Genre-based features
    if genres_col in df.columns:
        print("Computing genre-based features...")

        # genre_count
        df['genre_count'] = df[genres_col].apply(
            lambda x: len(x) if isinstance(x, (list, set)) else 0
        )
        df['has_genres'] = (df['genre_count'] > 0).astype(int)

        # Compute top N genres
        all_genres = (
            df[genres_col]
            .dropna()
            .apply(lambda x: x if isinstance(x, (list, set)) else [])
            .explode()
        )

        top_genres = set(all_genres.value_counts().head(top_n_genres).index)

        # is_top_genre -> 1 if the book has any genre in the top list
        df['is_top_genre'] = df[genres_col].apply(
            lambda xs: int(any(g in top_genres for g in (xs or [])))
            if isinstance(xs, (list, set)) else 0
        )
        print(f"Completed genre-based features"
              f" using top {top_n_genres} genres.")

    # Author popularity proxy
    if author_col in df.columns:
        print("Computing author-based features...")
        author_counts = df[author_col].value_counts()

        df['author_book_count'] = df[author_col].map(author_counts).fillna(0)

        # Determine top N authors
        top_authors = set(author_counts.head(top_n_authors).index)

        df['is_top_author'] = df[author_col].apply(
            lambda a: int(a in top_authors)
        )
        print(f"Completed author-based features"
              f" using top {top_n_authors} authors.")

        # Add primary_author feature
        df['primary_author'] = df[author_col].apply(extract_primary_author)
        print("Extracted primary_author feature.")

    # Publisher popularity proxy
    if publisher_col in df.columns:
        print("Computing publisher-based features...")
        publisher_counts = df[publisher_col].value_counts()
        df['publisher_book_count'] = (
            df[publisher_col].map(publisher_counts).fillna(0)
            )
        print("Completed publisher-based features.")

    # Series metadata simplification
    if series_col in df.columns:
        print("Computing series-based features...")
        df['in_series'] = df[series_col].notna().astype(int)
        print("Completed series-based features.")

    # Textual metadata
    if text_col in df.columns:
        print("Computing textual metadata features...")
        df['description_length'] = df[text_col].apply(
            lambda x: len(str(x)) if isinstance(x, str) else 0
        )
        df['description_word_count'] = df[text_col].apply(
            lambda x: len(str(x).split()) if isinstance(x, str) else 0
        )
        print("Completed textual metadata features.")

    # Text embeddings
    if encode_text_embeddings and text_col in df.columns:
        if ST_MODEL is None:
            raise ImportError(
                "SentenceTransformer not available. "
                "Install it or set encode_text_embeddings=False."
            )

        print("Computing text embeddings (this may take a while)...")
        df['text_embedding'] = df[text_col].apply(
            lambda x: ST_MODEL.encode(str(x))
            if isinstance(x, str)
            else np.zeros(384)
        )
        print("Completed text embeddings.")

        print("Computing scalar features from embeddings...")
        # Derive scalar features from embeddings for correlation/PPS analysis
        df['embedding_l2_norm'] = df['text_embedding'].apply(
            lambda x: np.linalg.norm(x)
            if isinstance(x, np.ndarray)
            else np.nan
        )
        df['embedding_mean'] = df['text_embedding'].apply(
            lambda x: x.mean() if isinstance(x, np.ndarray) else np.nan
        )
        df['embedding_std'] = df['text_embedding'].apply(
            lambda x: x.std() if isinstance(x, np.ndarray) else np.nan
        )
        print("Completed scalar features from embeddings.")

    print("Feature engineering complete.\n")
    return df


def extract_primary_author(author_string):
    """Extract the first author from a comma-separated author string."""
    if pd.isna(author_string):
        return None
    # Ensure it is str
    author_string = str(author_string)
    # Split by comma and return first element
    primary = author_string.split(',', maxsplit=1)[0].strip()
    return primary if primary else None
