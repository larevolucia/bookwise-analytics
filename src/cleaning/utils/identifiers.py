""" Identifier detection and cleaning functions. """
import re
import pandas as pd
import numpy as np


def detect_asin(x):
    """Detect if the input is an ASIN (10-char Amazon ID) rather than an ISBN.

    Returns True for likely ASINs, False otherwise.
    """
    if pd.isna(x):
        return False
    s = str(x).strip().upper()

    # If it's a pure 13-digit number, that's ISBN-13 (not ASIN)
    if re.fullmatch(r'\d{13}', s):
        return False

    # If it matches ISBN-10 formats (9 digits + X or 10 digits),
    # treat as ISBN (not ASIN)
    if re.fullmatch(r'\d{9}X', s) or re.fullmatch(r'\d{10}', s):
        return False

    # ASINs are 10-character alphanumeric codes and commonly contain letters.
    # Only flag as ASIN if it's 10 chars and contains a letter
    if re.fullmatch(r'[A-Z0-9]{10}', s) and re.search(r'[A-Z]', s):
        return True

    return False


def clean_isbn(raw):
    """
    Clean and standardize ISBN values from messy inputs.

    Args:
        raw: Raw ISBN string or number

    Returns:
        Cleaned ISBN-10 or ISBN-13, or np.nan if invalid
    """
    if pd.isna(raw):
        return np.nan

    raw = str(raw).strip().upper()

    # Detect ASIN (must check before cleaning)
    if detect_asin(raw):
        return np.nan

    # Fix scientific notation
    if 'e' in raw.lower():
        try:
            raw = '{:.0f}'.format(float(raw))
        except (ValueError, TypeError, OverflowError):
            return np.nan

    # Keep X for ISBN-10, remove all other non-digits
    cleaned = re.sub(r'[^0-9X]', '', raw)

    # Remove known placeholders
    if cleaned in [
        '0000000000',
        '0000000000000',
        '9999999999',
        '9999999999999'
    ]:
        return np.nan

    # Handle different lengths
    length = len(cleaned)

    # ISBN-13 must be exactly 13 digits (no X allowed)
    if length == 13:
        if cleaned.isdigit():
            return cleaned
        else:
            return np.nan  # Has 'X' or other issues

    # ISBN-10 with X must have X at the end only
    if length == 10:
        # Reject 10-digit numbers that start with ISBN-13 prefixes
        if cleaned.isdigit() and cleaned.startswith(('977', '978', '979')):
            return np.nan
        if cleaned.isdigit():
            return cleaned
        elif cleaned[-1] == 'X' and cleaned[:-1].isdigit():
            return cleaned
        else:
            return np.nan

    # Handle 9-digit ISBN-10 (missing leading zero) - pad with zero
    # But ONLY if it doesn't look like a truncated ISBN-13
    if length == 9:
        # If starts with 977, 978, 979, it's likely a truncated ISBN-13
        if cleaned.startswith(('977', '978', '979')):
            return np.nan
        # Otherwise, assume it's a valid ISBN-10 missing leading zero
        if cleaned.isdigit():
            return '0' + cleaned
        else:
            return np.nan

    # All other lengths are invalid
    return np.nan
