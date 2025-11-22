""" Language cleaning utilities.
"""
import json
import numpy as np
from .helpers import _safe_strip_lower, _is_empty


def load_language_mapping():
    """Load the language mapping dictionary."""
    with open(
        "src/cleaning/mappings/languages_dict.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def clean_language_field(value, lang_dict=None):
    """
    Standardize language codes to ISO 639-1 format.

    Args:
        value: Raw language string or code
        lang_dict: Optional pre-loaded language mapping dict

    Returns:
        Standardized 2-letter language code or NaN
    """
    if _is_empty(value):
        return np.nan

    if lang_dict is None:
        lang_dict = load_language_mapping()

    # clean and standardize
    value = _safe_strip_lower(value)

    # handle regional variants (en-US -> en)
    if value and '-' in value:
        value = value.split('-', maxsplit=1)[0]

    # map to standard code
    mapped = lang_dict.get(value)

    # if mapping fails, return NaN
    return mapped if mapped is not None else np.nan
