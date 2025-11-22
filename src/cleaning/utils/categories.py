""" Category cleaning utilities """
import ast
import re
import pandas as pd
import numpy as np
from .helpers import _deduplicate_list


def parse_list_field(val):
    """
    Safely parse a stringified list (e.g. '["x", "y"]') into a Python list.
    Returns np.nan for missing, invalid, or empty values.

    Args:
        val: Stringified list or actual list

    Returns:
        Cleaned list or np.nan
    """
    if pd.isna(val) or val in ['[]', '', None, 'None']:
        return np.nan
    try:
        lst = ast.literal_eval(str(val))
        if isinstance(lst, list):
            cleaned = [
                str(x).strip()
                for x in lst
                if isinstance(x, str) and x.strip()
            ]
            return cleaned or np.nan
    except (ValueError, SyntaxError, TypeError):
        return np.nan
    return np.nan


def clean_genre_list(subjects_list):
    """
    Clean and normalize a list of genres/subjects from APIs.

    Args:
        subjects_list: List of raw subject strings

    Returns:
        List of cleaned, lowercase genre strings or None
    """
    if not isinstance(subjects_list, list) or len(subjects_list) == 0:
        return None

    cleaned = []
    for subject in subjects_list:
        if not isinstance(subject, str):
            continue

        # Basic cleaning
        clean = subject.lower().strip()

        # Remove common noise patterns
        clean = re.sub(r'\s*\(.*?\)\s*', '', clean)  # Remove parentheticals
        clean = re.sub(r'[^a-z0-9\s\-]', '', clean)  # Remove special chars
        clean = re.sub(r'\s+', ' ', clean).strip()

        if clean and len(clean) > 2:  # Skip very short entries
            cleaned.append(clean)

    # Deduplicate while preserving order
    unique = _deduplicate_list(cleaned)

    return unique if unique else None


def clean_post_parsing(genres, *lists_to_remove):
    """
    Remove known non-genre or auxiliary tags safely.
    Accepts any number of lists/sets to remove.

    Args:
        genres: List of genre strings
        *lists_to_remove: Variable number of lists containing terms to remove

    Returns:
        Filtered genre list
    """
    if not isinstance(genres, list):
        return genres

    # Combine all removal terms into one flat set
    removal_terms = set().union(*lists_to_remove)

    return [g for g in genres if g.lower().strip() not in removal_terms]
