"""
Unit tests for cleaning_utils.py
"""
import pandas as pd
import numpy as np

from src.cleaning.utils.categories import clean_genre_list
from src.cleaning.utils.dates import clean_date_string, parse_mixed_date
from src.cleaning.utils.identifiers import clean_isbn
from src.cleaning.utils.language import clean_language_field
from src.cleaning.utils.text_cleaning import clean_title
from src.cleaning.utils.numeric import clean_pages_field


class TestTitleCleaning:
    """Tests for cleaning title field"""
    def test_clean_title_basic(self):
        """Apostrophes are preserved in the cleaning function"""
        assert clean_title("Harry Potter and the Philosopher's Stone") == (
            "harry potter and the philosopher's stone"
        )

    def test_clean_title_with_series(self):
        """ Parenthetical series info is removed """
        assert clean_title("The Hunger Games (Book 1)") == "the hunger games"

    def test_clean_title_with_edition(self):
        """ Edition info in brackets is removed """
        assert clean_title("1984 [Illustrated Edition]") == "1984"

    def test_clean_title_nan(self):
        """ NaN input returns NaN """
        result = clean_title(np.nan)
        assert pd.isna(result)


class TestISBNCleaning:
    """Test ISBN cleaning with real examples from the datasets."""

    def test_clean_isbn_valid_13_hyphenated(self):
        """Test ISBN-13 with hyphens"""
        assert clean_isbn("978-0-439-70818-4") == "9780439708184"

    def test_clean_isbn_valid_13_no_hyphens(self):
        """Test ISBN-13 without hyphens - real example from BBE"""
        assert clean_isbn("9780439023481") == "9780439023481"

    def test_clean_isbn_9_digit_padded(self):
        """Test real 9-digit ISBN-10 from Goodbooks (missing leading zero)"""
        # Real examples that don't start with 977/978/979
        assert clean_isbn("439023483") == "0439023483"
        assert clean_isbn("439554934") == "0439554934"
        assert clean_isbn("316015849") == "0316015849"

    def test_clean_isbn_valid_10_with_x(self):
        """Test real ISBN-10 ending in X from Goodbooks"""
        assert clean_isbn("043965548X") == "043965548X"
        assert clean_isbn("0-439-65548-X") == "043965548X"

    def test_clean_isbn_valid_10_all_digits(self):
        """Test ISBN-10 with all digits"""
        assert clean_isbn("0439023483") == "0439023483"
        assert clean_isbn("1788441028") == "1788441028"  # Real from BBE

    def test_clean_isbn_placeholder_rejected(self):
        """Test placeholder ISBNs get rejected - real examples from BBE"""
        assert pd.isna(clean_isbn("9999999999999"))
        assert pd.isna(clean_isbn("0000000000000"))
        assert pd.isna(clean_isbn("9999999999"))

    def test_clean_isbn_asin_rejected(self):
        """Test ASIN gets rejected"""
        assert pd.isna(clean_isbn("B00ABC1234"))

    def test_clean_isbn_truncated_isbn13_rejected(self):
        """Test truncated ISBN-13 gets rejected"""
        # starts with 977/978/979 but only 10 digits
        assert pd.isna(clean_isbn("9780615801"))  # Truncated, starts with 978
        assert pd.isna(clean_isbn("9770916099"))  # Truncated, starts with 977
        assert pd.isna(clean_isbn("9781452439"))  # Truncated, starts with 978

    def test_clean_isbn_truncated_isbn13_9_digits_rejected(self):
        """Test 9-digit truncated ISBN-13 gets rejected"""
        # These start with 977/978/979 so likely truncated ISBN-13
        assert pd.isna(clean_isbn("977091609"))
        assert pd.isna(clean_isbn("978061580"))

    def test_clean_isbn_invalid_length_rejected(self):
        """Test invalid length ISBNs get rejected"""
        assert pd.isna(clean_isbn("12345"))
        assert pd.isna(clean_isbn("123456"))
        assert pd.isna(clean_isbn("12345678"))
        assert pd.isna(clean_isbn("12345678901"))  # 11 digits
        assert pd.isna(clean_isbn("123456789012"))  # 12 digits

    def test_clean_isbn_empty_and_nan(self):
        """Test empty strings and NaN values"""
        assert pd.isna(clean_isbn(""))
        assert pd.isna(clean_isbn(np.nan))
        assert pd.isna(clean_isbn(None))

    def test_clean_isbn_with_spaces(self):
        """Test ISBN with spaces"""
        assert clean_isbn("978 0439 023481") == "9780439023481"
        assert clean_isbn("0 439 65548 X") == "043965548X"

    def test_clean_isbn_multiple_real_examples_from_bbe(self):
        """Test multiple real ISBNs from BBE dataset"""
        assert clean_isbn("9780316015844") == "9780316015844"
        assert clean_isbn("9780375831003") == "9780375831003"
        assert clean_isbn("9780451526342") == "9780451526342"
        assert clean_isbn("1788441028") == "1788441028"
        assert clean_isbn("0578530805") == "0578530805"

    def test_clean_isbn_multiple_real_examples_from_goodbooks(self):
        """Test multiple real ISBNs from Goodbooks original dataset"""
        assert clean_isbn("439023483") == "0439023483"  # 9 digits, padded
        assert clean_isbn("439554934") == "0439554934"  # 9 digits, padded
        assert clean_isbn("316015849") == "0316015849"  # 9 digits, padded
        assert clean_isbn("043965548X") == "043965548X"  # With X


class TestLanguageCleaning:
    """Tests for cleaning language field"""
    def test_clean_language_standard(self):
        """ Standard language name """
        assert clean_language_field("English") == "en"

    def test_clean_language_regional(self):
        """ Regional language code """
        assert clean_language_field("en-US") == "en"

    def test_clean_language_unknown(self):
        """ Unknown or missing language """
        assert clean_language_field("") is np.nan
        assert clean_language_field(np.nan) is np.nan


class TestDateCleaning:
    """Tests for cleaning publication date field"""
    def test_clean_date_string_removes_ordinals(self):
        """Remove ordinal suffixes from dates"""
        assert clean_date_string("April 27th, 2010") == "April 27, 2010"
        assert clean_date_string("December 1st, 2005") == "December 1, 2005"
        assert clean_date_string("March 22nd, 2015") == "March 22, 2015"
        assert clean_date_string("May 3rd, 2020") == "May 3, 2020"

    def test_clean_date_string_with_nan(self):
        """NaN input returns NaN"""
        result = clean_date_string(np.nan)
        assert pd.isna(result)

    def test_parse_mixed_date_full_date(self):
        """Parse full date strings"""
        result = parse_mixed_date("April 27, 2010")
        expected = pd.Timestamp("2010-04-27")
        assert result == expected

    def test_parse_mixed_date_year_only(self):
        """Parse year-only dates"""
        result = parse_mixed_date("2010")
        expected = pd.Timestamp("2010-01-01")
        assert result == expected

    def test_parse_mixed_date_various_formats(self):
        """Parse various date formats using fuzzy parsing"""
        assert parse_mixed_date("2010-04-27") == pd.Timestamp("2010-04-27")
        assert parse_mixed_date("04/27/2010") == pd.Timestamp("2010-04-27")
        assert parse_mixed_date("27 Apr 2010") == pd.Timestamp("2010-04-27")

    def test_parse_mixed_date_with_ordinals_combined(self):
        """Test combined workflow: clean then parse"""
        raw = "April 27th, 2010"
        cleaned = clean_date_string(raw)
        result = parse_mixed_date(cleaned)
        expected = pd.Timestamp("2010-04-27")
        assert result == expected

    def test_parse_mixed_date_invalid(self):
        """Invalid dates return NaN"""
        assert pd.isna(parse_mixed_date("not a date"))
        assert pd.isna(parse_mixed_date(""))
        assert pd.isna(parse_mixed_date(np.nan))

    def test_parse_mixed_date_empty_string(self):
        """Empty string returns NaN"""
        result = parse_mixed_date("")
        assert pd.isna(result)

    def test_parse_mixed_date_nan(self):
        """NaN input returns NaN"""
        result = parse_mixed_date(np.nan)
        assert pd.isna(result)


class TestPagesCleaning:
    """Tests for cleaning number of pages field"""
    def test_clean_pages_valid(self):
        """ Valid number of pages """
        assert clean_pages_field(350) == 350
        assert clean_pages_field("350 pages") == 350

    def test_clean_pages_out_of_range(self):
        """ Out of range number of pages """
        assert clean_pages_field(5) is None  # Too few
        assert clean_pages_field(5000) is None  # Too many

    def test_clean_pages_invalid(self):
        """ Invalid number of pages """
        assert clean_pages_field("invalid") is None
        assert clean_pages_field(np.nan) is None


class TestGenreCleaning:
    """Tests for genre list cleaning"""
    def test_clean_genre_list_basic(self):
        """ Basic genre list """
        result = clean_genre_list(["Fiction", "Fantasy", "Young Adult"])
        assert "fiction" in result
        assert "fantasy" in result
        assert "young adult" in result

    def test_clean_genre_list_with_noise(self):
        """ Genre list with noise """
        result = clean_genre_list(["Fiction (General)", "Fantasy & Magic"])
        assert "fiction general" in result or "fiction" in result

    def test_clean_genre_list_empty(self):
        """ Empty genre list """
        assert clean_genre_list([]) is None
        assert clean_genre_list(None) is None
