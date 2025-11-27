""" Utility functions for cleaning and parsing date strings."""
import re
import pandas as pd
import numpy as np


def parse_mixed_date(date_str):
    """
    Parse dates with proper 2-digit year handling for historical books.
    Assumes 2-digit years 00-40 are 2000s, 41-99 are 1900s.
    """
    if pd.isna(date_str) or date_str == '':
        return pd.NaT

    try:
        # First attempt: parse the date
        parsed = pd.to_datetime(date_str, errors='coerce')

        if pd.notna(parsed):
            # Check if year is suspiciously
            # recent (likely a 2-digit year issue)
            if parsed.year > 2020:
                # Extract original year string to check if it was 2-digit
                year_match = re.search(r'\b(\d{2})\b', str(date_str))
                if year_match:
                    two_digit_year = int(year_match.group(1))
                    # Adjust: 00-40 → 2000s, 41-99 → 1900s
                    if two_digit_year <= 40:
                        century = 2000
                    else:
                        century = 1900
                    # Reconstruct with correct century
                    correct_year = century + two_digit_year
                    parsed = parsed.replace(year=correct_year)

            return parsed

    except ValueError:
        return pd.NaT

    return pd.NaT


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


def extract_year(date_str):
    """Extract year from date string."""
    if pd.isna(date_str):
        return np.nan

    date_str = str(date_str).strip()

    # Try to extract 4-digit year
    year_match = re.search(r'\b(1[5-9]\d{2}|20[0-2]\d)\b', date_str)

    if year_match:
        year = int(year_match.group(1))
        # Validate reasonable range
        if 1800 <= year <= 2025:
            return year

    return np.nan
