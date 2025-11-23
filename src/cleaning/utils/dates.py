""" Utility functions for cleaning and parsing date strings."""
import re
import pandas as pd
import numpy as np

ORDINAL_SUFFIX_PATTERN = r'(\d+)(st|nd|rd|th)\b'


def parse_mixed_date(date_str):
    """ Try to parse a variety of date formats safely."""
    if pd.isna(date_str) or date_str == "":
        return np.nan

    try:
        # Use dayfirst=False to prioritize MM/DD/YY format
        parsed = pd.to_datetime(
            date_str,
            format='mixed',
            dayfirst=False,
            errors='coerce'
        )

        # Fix two-digit year ambiguity (assume 1900s for years > current year)
        if pd.notna(parsed) and parsed.year > 2025:
            parsed = parsed.replace(year=parsed.year - 100)

        return parsed
    except (ValueError, TypeError):
        return np.nan


def clean_date_string(date_str):
    """Remove ordinal suffixes and unwanted characters from a date string."""
    if pd.isna(date_str):
        return np.nan
    # remove st, nd, rd, th (like 'April 27th 2010' → 'April 27 2010')
    cleaned = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', str(date_str))
    return cleaned.strip()


def format_date_iso(date_obj):
    """
    Convert a datetime object to ISO format string (YYYY-MM-DD).
    Returns np.nan for invalid inputs.
    """
    if pd.isna(date_obj):
        return np.nan

    try:
        if isinstance(date_obj, pd.Timestamp):
            return date_obj.strftime("%Y-%m-%d")
        elif isinstance(date_obj, str):
            # If already a string, try to parse and reformat
            parsed = pd.to_datetime(date_obj, errors='coerce')
            if pd.notna(parsed):
                return parsed.strftime("%Y-%m-%d")
        return np.nan
    except (ValueError, AttributeError):
        return np.nan
