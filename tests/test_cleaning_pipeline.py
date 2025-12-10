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

    @pytest.fixture
    def sample_publisher_data(self):
        """Sample data with various publisher formats"""
        return pd.DataFrame({
            'book_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'title_clean': [
                'Book 1',
                'Book 2',
                'Book 3',
                'Book 4',
                'Book 5',
                'Book 6',
                'Book 7',
                'Book 8'
            ],
            'publisher_openlib': [
                'HarperCollins Publishers',
                '  Penguin Books  ',
                'Random House.',
                '"William Morrow"',
                'Tor Books',
                '123',  # numeric only
                None,
                'Simon & Schuster'
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

        # check that dates are formatted as ISO strings
        assert result.loc[0, 'publication_date_openlib_clean'] == '1949-01-01'

        date_1 = result.loc[1, 'publication_date_openlib_clean']
        assert '1937-09-21' == date_1

        date_3 = result.loc[3, 'publication_date_openlib_clean']
        assert '2020-01-15' == date_3

        # invalid dates should be NaN
        assert pd.isna(result.loc[2, 'publication_date_openlib_clean'])
        assert pd.isna(result.loc[4, 'publication_date_openlib_clean'])

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

    def test_clean_openlibrary_publisher(self, sample_publisher_data):
        """Test that OpenLibrary publishers are cleaned and consolidated"""
        result = apply_cleaners_selectively(
            sample_publisher_data,
            fields_to_clean=['publisher'],
            source_suffix='_openlib',
            target_suffix='_openlib_clean',
            inplace=False
        )

        assert 'publisher_openlib_clean' in result.columns

        # Test parent company consolidation
        # (with apply_parent_mapping=True by default)
        assert result.loc[0, 'publisher_openlib_clean'] == 'harpercollins'
        assert result.loc[1, 'publisher_openlib_clean'] == (
            'penguin random house'
        )
        assert result.loc[2, 'publisher_openlib_clean'] == (
            'penguin random house'
        )
        assert result.loc[3, 'publisher_openlib_clean'] == 'harpercollins'
        # Test unmapped publisher (should pass through cleaned)
        assert result.loc[4, 'publisher_openlib_clean'] == 'tor books'

        # Test numeric-only publisher (should be None)
        assert pd.isna(result.loc[5, 'publisher_openlib_clean'])

        # Test missing value
        assert pd.isna(result.loc[6, 'publisher_openlib_clean'])

        # Test ampersand removal in Simon & Schuster
        result_7 = result.loc[7, 'publisher_openlib_clean']
        assert 'simon' in result_7 and 'schuster' in result_7

    def test_clean_multiple_fields_together(self, sample_openlibrary_data):
        """Test cleaning multiple fields in a single pipeline call"""
        result = apply_cleaners_selectively(
            sample_openlibrary_data,
            fields_to_clean=[
                'pages',
                'publication_date',
                'language',
                'subjects'
                ],
            source_suffix='_openlib',
            target_suffix='_openlib_clean',
            inplace=False
        )

        # Verify all cleaned columns were created
        expected_columns = [
            'pages_openlib_clean',
            'publication_date_openlib_clean',
            'language_openlib_clean',
            'subjects_openlib_clean'
        ]
        for col in expected_columns:
            assert col in result.columns

        # Verify original columns are still present
        assert 'pages_openlib' in result.columns
        assert 'publication_date_openlib' in result.columns
        assert 'language_openlib' in result.columns
        assert 'subjects_openlib' in result.columns

    def test_inplace_false_preserves_original(self, sample_openlibrary_data):
        """Test that inplace=False doesn't modify original DataFrame"""
        original_columns = sample_openlibrary_data.columns.tolist()

        result = apply_cleaners_selectively(
            sample_openlibrary_data,
            fields_to_clean=['pages'],
            source_suffix='_openlib',
            target_suffix='_openlib_clean',
            inplace=False
        )

        # Original DataFrame should be unchanged
        assert sample_openlibrary_data.columns.tolist() == original_columns
        assert 'pages_openlib_clean' not in sample_openlibrary_data.columns

        # Result should have the new column
        assert 'pages_openlib_clean' in result.columns
