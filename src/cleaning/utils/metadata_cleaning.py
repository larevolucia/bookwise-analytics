""" Metadata cleaning utilities."""
import re
import pandas as pd
import numpy as np
from .helpers import normalize_unicode

SERIES_ORDER_PATTERN = r'(#\d+|book\s*\d+|volume\s*\d+|vol\.\s*\d+)'


def clean_and_split_authors(name):
    """
    Cleans author names and returns a list of authors.

    Args:
        name: Raw author name string

    Returns:
        List of cleaned author names or None
    """
    if pd.isna(name):
        return None

    # Remove role descriptors
    cleaned = re.sub(r"\s*\([^)]*\)", "", name)

    # Split on commas and lowercase each name
    authors_list = [a.strip().lower() for a in cleaned.split(",") if a.strip()]

    # Normalize each name
    normalized = [normalize_unicode(a) for a in authors_list]

    # Drop empty results
    normalized = [a for a in normalized if isinstance(a, str) and a]

    return normalized if normalized else None


def clean_series(series: str) -> str:
    """
    Clean book series names for consistency.

    Steps:
    - Remove bracketed or parenthetical notes (e.g. '(Book 3)', '[Series]')
    - Remove numeric indicators (#1, Book 2) while preserving base name
    - Normalize spaces and lowercase

    Args:
        series: Raw series string

    Returns:
        Cleaned series name or np.nan
    """
    if not isinstance(series, str) or series.strip() == "":
        return np.nan

    text = normalize_unicode(series)
    text = text.strip()

    # Remove bracketed or parenthetical content
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)

    # Remove series order markers (e.g. "#1", "Book 2", "Vol. 3")
    text = re.sub(
        SERIES_ORDER_PATTERN,
        "",
        text,
        flags=re.IGNORECASE
    )

    # Clean punctuation and normalize spacing
    text = re.sub(r"[^\w\s'\-]", "", text)
    text = re.sub(r"\s+", " ", text)

    # Remove leftover "-3" or "—3"
    text = re.sub(r"[-—]\s*\d+\b", "", text)

    # Remove trailing standalone numbers (e.g., "narnia 7")
    text = re.sub(r"\b\d+\b$", "", text)

    return text.strip().lower()


def assign_parent(pub, publisher_patterns):
    """Map publisher names to parent companies based on patterns."""
    if pd.isna(pub):
        return pub
    for parent, kws in publisher_patterns.items():
        for kw in kws:
            if kw in pub:
                return parent
    return pub


def clean_awards_list(lst):
    """
    Clean and normalize awards list:
    - Lowercase and remove punctuation noise
    - Remove year patterns like (2009) or (2010)
    - Deduplicate entries

    Args:
        lst: List of award strings

    Returns:
        Cleaned and deduplicated list or np.nan
    """
    if not isinstance(lst, list):
        return np.nan

    cleaned = []
    for a in lst:
        if not isinstance(a, str) or not a.strip():
            continue
        a = a.lower().strip()
        # Remove (YYYY) patterns
        a = re.sub(r'\(\s*\d{4}\s*\)', '', a)
        # Remove leftover punctuation and extra spaces
        a = re.sub(r'[^a-z0-9\s\-\&\'"]', '', a)
        a = re.sub(r'\s+', ' ', a).strip()
        cleaned.append(a)

    cleaned = list(set(cleaned))
    return cleaned if cleaned else np.nan
