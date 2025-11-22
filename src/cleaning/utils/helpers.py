""" Helper functions for data cleaning."""
import unicodedata
import re
import pandas as pd
import numpy as np


def _is_empty(value):
    """Check if value is effectively empty."""
    value = str(value).strip().lower()
    return pd.isna(value) or value in ['', 'nan', 'None', '[]', 'unknown']


def _safe_strip_lower(value):
    """Safely convert to lowercase stripped string."""
    if pd.isna(value):
        return None
    return str(value).strip().lower()


def _deduplicate_list(items):
    """Deduplicate list while preserving order."""
    seen = set()
    return [x for x in items if not (x in seen or seen.add(x))]


def normalize_unicode(s):
    """Normalize unicode characters in a string."""
    if not isinstance(s, str):
        return np.nan
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("'", "'")  # Fix common apostrophe issue
    s = re.sub(r"[^\w\s'\-]", "", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s
