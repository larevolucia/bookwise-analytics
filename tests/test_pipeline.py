""" Tests for cleaning pipeline utilities """
import pytest
import pandas as pd
from src.cleaning.utils.pipeline import apply_cleaners_selectively


class TestApplyCleanersSelectively:
    """Test suite for the apply_cleaners_selectively function"""
    @pytest.fixture
    def sample_openlibrary_data(self):
        """Sample data mimicking OpenLibrary API responses"""
        return pd.DataFrame({
            'book_id': [1, 2, 3, 4, 5],
            'title_clean': [
                '1984',
                'The Hobbit',
                'Pride and Prejudice',
                'Unknown Book',
                'Test'
                ],
            # OpenLibrary API fields
            'pages_openlib': [328, 310, None, 'invalid', 250],
            'publication_date_openlib': [
                '1949',
                'September 21, 1937',
                None,
                '2020-01-15',
                'invalid'
                ],
            'language_openlib': ['eng', 'en', None, 'fre', 'unknown'],
            'subjects_openlib': [
                ['dystopia', 'science fiction', 'Classic'],
                ['fantasy', 'Adventure'],
                None,
                ['Romance', 'Historical Fiction'],
                []
            ]
        })

    def test_clean_openlibrary_pages(self, sample_openlibrary_data):
        """Test that OpenLibrary page counts are cleaned correctly"""
        result = apply_cleaners_selectively(
            sample_openlibrary_data,
            fields_to_clean=['pages'],
            source_suffix='_openlib',
            target_suffix='_openlib_clean',
            inplace=False
        )

        # Check that cleaned column was created
        assert 'pages_openlib_clean' in result.columns

        # Valid page counts should be preserved
        assert result.loc[0, 'pages_openlib_clean'] == 328
        assert result.loc[1, 'pages_openlib_clean'] == 310
        assert result.loc[4, 'pages_openlib_clean'] == 250

        # Invalid/missing values should be None
        assert pd.isna(result.loc[2, 'pages_openlib_clean'])
        assert pd.isna(result.loc[3, 'pages_openlib_clean'])

    def test_clean_openlibrary_dates(self, sample_openlibrary_data):
        """Test that OpenLibrary publication dates are parsed correctly"""
        result = apply_cleaners_selectively(
            sample_openlibrary_data,
            fields_to_clean=['publication_date'],
            source_suffix='_openlib',
            target_suffix='_openlib_clean',
            inplace=False
        )

        assert 'publication_date_openlib_clean' in result.columns

        # Simple year should be parsed
        date_0 = result.loc[0, 'publication_date_openlib_clean']
        assert date_0 is not None
        assert '1949' in str(date_0)

        # Full date should be parsed
        date_1 = result.loc[1, 'publication_date_openlib_clean']
        assert date_1 is not None
        assert '1937' in str(date_1)

        # ISO date should work
        date_3 = result.loc[3, 'publication_date_openlib_clean']
        assert date_3 is not None
        assert '2020' in str(date_3)

    def test_clean_openlibrary_language(self, sample_openlibrary_data):
        """Test that OpenLibrary language codes are standardized"""
        result = apply_cleaners_selectively(
            sample_openlibrary_data,
            fields_to_clean=['language'],
            source_suffix='_openlib',
            target_suffix='_openlib_clean',
            inplace=False
        )

        assert 'language_openlib_clean' in result.columns

        # 'eng' should map to 'en'
        assert result.loc[0, 'language_openlib_clean'] == 'en'

        # 'en' should stay 'en'
        assert result.loc[1, 'language_openlib_clean'] == 'en'

        # 'fre' should map to 'fr'
        assert result.loc[3, 'language_openlib_clean'] == 'fr'

        # Missing/unknown should be NaN
        assert pd.isna(result.loc[2, 'language_openlib_clean'])

    def test_clean_openlibrary_subjects(self, sample_openlibrary_data):
        """Test that OpenLibrary subjects/genres are cleaned"""
        result = apply_cleaners_selectively(
            sample_openlibrary_data,
            fields_to_clean=['subjects'],
            source_suffix='_openlib',
            target_suffix='_openlib_clean',
            inplace=False
        )

        assert 'subjects_openlib_clean' in result.columns

        # Check first book's genres (should be lowercase and cleaned)
        genres_0 = result.loc[0, 'subjects_openlib_clean']
        assert isinstance(genres_0, list)
        assert 'dystopia' in genres_0
        assert 'science fiction' in genres_0
        assert 'classic' in genres_0

        # Check second book
        genres_1 = result.loc[1, 'subjects_openlib_clean']
        assert 'fantasy' in genres_1
        assert 'adventure' in genres_1

        # Empty or None should return None
        assert result.loc[2, 'subjects_openlib_clean'] is None
        assert result.loc[4, 'subjects_openlib_clean'] is None
