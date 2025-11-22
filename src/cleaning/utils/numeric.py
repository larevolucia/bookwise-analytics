""" Numeric cleaning utilities """
import re
import numpy as np
import pandas as pd


NUMERIC_EXTRACTION_PATTERN = r'(\d+)'
# ==========================================
# PAGES CLEANING
# ==========================================


def clean_pages_field(value, min_pages=10, max_pages=3000):
    """
    Validate and clean page count values.

    Args:
        value: Raw page count (int, float, or string)
        min_pages: Minimum valid page count
        max_pages: Maximum valid page count

    Returns:
        Cleaned integer page count or None
    """
    if pd.isna(value):
        return None

    try:
        # Extract numeric value
        if isinstance(value, str):
            match = re.search(NUMERIC_EXTRACTION_PATTERN, value)
            if match:
                value = match.group(1)
            else:
                return None

        pages = int(float(value))

        # Validate range
        if min_pages <= pages <= max_pages:
            return pages
        else:
            return None

    except (ValueError, TypeError):
        return None


# ==========================================
# PRICE CLEANING
# ==========================================

def clean_price(value):
    """
    Standardize price values by handling missing, textual, and range inputs.

    Args:
        value: Raw price value

    Returns:
        Cleaned float price or np.nan
    """
    if pd.isna(value):
        return np.nan

    s = str(value).strip().lower()

    # Handle common invalid or missing cases
    if s in {'free', 'none', 'nan', '', 'n/a', '-', '—'}:
        return np.nan

    # Remove non-numeric symbols
    s = re.sub(r'[^\d.,\-]', '', s)

    # Handle ranges like '12-15' → mean of both
    if '-' in s:
        try:
            nums = [float(x) for x in re.split(r'[-—]', s) if x.strip()]
            return np.mean(nums) if nums else np.nan
        except ValueError:
            return np.nan

    # Convert to float, replacing commas with dots
    try:
        return float(s.replace(',', '.'))
    except ValueError:
        return np.nan
