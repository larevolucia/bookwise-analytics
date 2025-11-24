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


def map_subjects_to_genres(subjects):
    """
    Map OpenLibrary subjects to simplified genre categories.

    Args:
        subjects: List of subject strings from OpenLibrary

    Returns:
        List of simplified genre strings, or None if no matches found
    """
    if not isinstance(subjects, list) or len(subjects) == 0:
        return None

    # Define genre mapping keywords
    genre_mappings = {
        'fiction': ['fiction', 'novel', 'stories'],
        'fantasy': ['fantasy', 'magic', 'wizards', 'dragons'],
        'science fiction': [
            'science fiction',
            'sci-fi',
            'space',
            'futuristic'
            ],
        'mystery': ['mystery', 'detective', 'crime', 'thriller'],
        'romance': ['romance', 'love story', 'romantic'],
        'horror': ['horror', 'supernatural', 'ghost', 'vampire'],
        'historical fiction': ['historical fiction', 'history'],
        'biography': ['biography', 'autobiography', 'memoir'],
        'non-fiction': ['non-fiction', 'nonfiction'],
        'young adult': ['young adult', 'ya', 'teen', 'juvenile'],
        'children': ['children', 'juvenile', 'kids'],
        'poetry': ['poetry', 'poems'],
        'drama': ['drama', 'plays', 'theater'],
        'adventure': ['adventure', 'action'],
        'classics': ['classic', 'literature'],
        'humor': ['humor', 'comedy', 'satire'],
        'self-help': ['self-help', 'self help', 'motivational']
    }

    matched_genres = []
    subjects_lower = [s.lower() for s in subjects]

    # Check each genre mapping
    for genre, keywords in genre_mappings.items():
        for keyword in keywords:
            if any(keyword in subject for subject in subjects_lower):
                if genre not in matched_genres:
                    matched_genres.append(genre)
                break

    return matched_genres if matched_genres else None
