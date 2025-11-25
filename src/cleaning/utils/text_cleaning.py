""" Text cleaning utilities """
import unicodedata
import re
import html
import numpy as np
import pandas as pd
from .helpers import normalize_unicode


def clean_text_item(text, keep_pattern=r'[^a-z0-9\s-]'):
    """
    Lowercase and remove noise,
    keeping only letters, digits, hyphens and spaces.
    Can be reused for genres, awards, etc.

    Args:
        text: Raw text string
        keep_pattern: Regex pattern for characters to remove

    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ''
    text = text.lower().strip()
    text = re.sub(keep_pattern, '', text)  # Clean unwanted chars
    text = re.sub(r'\s+', ' ', text)       # Collapse multiple spaces
    return text.strip()


def clean_description(text):
    """
    Clean book description text from APIs.

    Args:
        text: Raw description string

    Returns:
        Cleaned description or None
    """
    if not isinstance(text, str):
        return None

    text = normalize_unicode(text)
    text = html.unescape(text)

    # Add a missing period and space between sentences when a lowercase letter
    # is immediately followed by an uppercase letter
    text = re.sub(r"([a-z])([A-Z])", r"\1. \2", text)

    # Remove librarian/editorial notes and metadata
    text = re.sub(r"\(Note:.*?\)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Librarian's note:.*?(?:\.)", "", text, flags=re.IGNORECASE)

    text = re.sub(
        r"Alternate cover edition of ISBN \d+",
        "",
        text,
        flags=re.IGNORECASE
    )
    text = re.sub(r"edition of ISBN \d+", "", text, flags=re.IGNORECASE)

    # Fix missing space after punctuation when followed by letter/number
    text = re.sub(r'([.,;!?])(?=[A-Za-z0-9])', r'\1 ', text)

    # Fix missing space after ISBN numbers (e.g., "9780679783268Since")
    text = re.sub(r'(\d)([A-Za-z])', r'\1 \2', text)

    # Normalize escaped quotes
    text = text.replace("\\'", "'").replace('\\"', '"')

    # Keep readable characters, include # for "#1"
    text = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ0-9#\s.,;!?\'\"-]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_title(title: str) -> str:
    """
    Clean book titles for consistency and API queries.

    Steps:
    - Remove bracketed or parenthetical notes (e.g. '(Box Set)', '[Hardcover]')
    - Remove common edition/format words
    - Remove ellipsis or truncated markers
    - Normalize spaces and lowercase

    Args:
        title: Raw title string

    Returns:
        Cleaned title string or np.nan
    """
    if not isinstance(title, str):
        return np.nan

    text = title.strip()

    # Normalize unicode characters
    text = unicodedata.normalize("NFKD", text)

    # Apostrophes standardization
    text = text.replace("'", "'")

    # Remove bracketed or parenthetical content
    text = re.sub(r"[\(\[].*?[\)\]]", "", text)

    # Remove common edition/format terms
    pattern = (
        r"\b("
        r"box set|collection|illustrated|edition|volume|vol\.|book\s*\d+"
        r")\b"
    )
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove ellipsis or truncation markers
    text = text.replace("...", "")

    # Remove extra punctuation and multiple spaces
    text = re.sub(r"[^\w\s'\-]", "", text)
    text = re.sub(r"\s+", " ", text)

    # Lowercase for consistency
    text = text.strip().lower()

    return text


def clean_description_nlp(text):
    """ Clean description text for NLP processing."""
    if pd.isna(text):
        return None
    text = normalize_unicode(text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()
