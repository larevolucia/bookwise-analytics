""" Tests for feature engineering functions """
import pytest
import pandas as pd
from src.modeling.feature_engineering import fe_engineering


class TestFeEngineering:
    """Test suite for the fe_engineering function"""

    @pytest.fixture
    def sample_data(self):
        """Sample data with various metadata fields"""
        return pd.DataFrame({
            'book_id': [1, 2, 3, 4, 5, 6, 7, 8],
            'description_clean': [
                'a classic dystopian novel about surveillance',
                'an epic fantasy adventure in middle earth',
                None,
                'a romance novel set in regency england',
                'short',
                'another dystopian story',
                'yet another fantasy tale',
                'more text here'
            ],
            'genres_clean': [
                ['dystopia', 'science fiction'],
                ['fantasy', 'adventure'],
                [],
                ['romance', 'historical'],
                None,
                ['dystopia', 'thriller'],
                ['fantasy', 'magic'],
                ['mystery', 'crime']
            ],
            'author_clean': [
                'george orwell',
                'j.r.r. tolkien',
                'george orwell',
                'jane austen',
                'unknown',
                'george orwell',
                'j.r.r. tolkien',
                'agatha christie'
            ],
            'publisher_clean': [
                'penguin random house',
                'harpercollins',
                'penguin random house',
                'penguin random house',
                'unknown',
                'penguin random house',
                'harpercollins',
                'small press'
            ],
            'series_clean': [
                None,
                'the lord of the rings',
                None,
                None,
                'test series',
                None,
                'the hobbit series',
                None
            ],
            'has_awards': [True, True, False, True, False, False, True, False],
            'is_major_publisher': [
                True, True, True, True, False, True, True, False
            ]
        })

    def test_genre_count_feature(self, sample_data):
        """Test that genre_count is calculated correctly"""
        result = fe_engineering(sample_data, encode_text_embeddings=False)

        assert 'genre_count' in result.columns
        assert result.loc[0, 'genre_count'] == 2
        assert result.loc[1, 'genre_count'] == 2
        assert result.loc[2, 'genre_count'] == 0
        assert result.loc[3, 'genre_count'] == 2
        assert result.loc[4, 'genre_count'] == 0

    def test_has_genres_feature(self, sample_data):
        """Test that has_genres binary flag is created"""
        result = fe_engineering(sample_data, encode_text_embeddings=False)

        assert 'has_genres' in result.columns
        assert result.loc[0, 'has_genres'] == 1
        assert result.loc[1, 'has_genres'] == 1
        assert result.loc[2, 'has_genres'] == 0
        assert result.loc[4, 'has_genres'] == 0

    def test_is_top_genre_feature(self, sample_data):
        """Test that is_top_genre identifies books with popular genres"""
        result = fe_engineering(
            sample_data,
            encode_text_embeddings=False,
            top_n_genres=3
        )

        assert 'is_top_genre' in result.columns
        # with top_n_genres=3, should be dystopia, fantasy
        # Book 0 has dystopia (appears 2 times), should be top genre
        assert result.loc[0, 'is_top_genre'] == 1
        # Book 1 has fantasy (appears 2 times), should be top genre
        assert result.loc[1, 'is_top_genre'] == 1
        # Book 2 has no genres
        assert result.loc[2, 'is_top_genre'] == 0
        # Book 4 has None
        assert result.loc[4, 'is_top_genre'] == 0

    def test_author_book_count_feature(self, sample_data):
        """Test that author popularity is calculated correctly"""
        result = fe_engineering(sample_data, encode_text_embeddings=False)

        assert 'author_book_count' in result.columns
        # George Orwell appears 3 times
        assert result.loc[0, 'author_book_count'] == 3
        assert result.loc[2, 'author_book_count'] == 3
        assert result.loc[5, 'author_book_count'] == 3
        # Tolkien appears twice
        assert result.loc[1, 'author_book_count'] == 2
        assert result.loc[6, 'author_book_count'] == 2
        # Others appear once
        assert result.loc[3, 'author_book_count'] == 1

    def test_is_top_author_feature(self, sample_data):
        """Test that is_top_author identifies popular authors"""
        result = fe_engineering(
            sample_data,
            encode_text_embeddings=False,
            top_n_authors=2
        )

        assert 'is_top_author' in result.columns
        # top_n_authors=2 -> George Orwell (3) and Tolkien (2)
        # George Orwell books
        assert result.loc[0, 'is_top_author'] == 1
        assert result.loc[2, 'is_top_author'] == 1
        assert result.loc[5, 'is_top_author'] == 1
        # Tolkien books
        assert result.loc[1, 'is_top_author'] == 1
        assert result.loc[6, 'is_top_author'] == 1
        # Non-top authors
        assert result.loc[3, 'is_top_author'] == 0
        assert result.loc[4, 'is_top_author'] == 0

    def test_publisher_book_count_feature(self, sample_data):
        """Test that publisher popularity is calculated correctly"""
        result = fe_engineering(sample_data, encode_text_embeddings=False)

        assert 'publisher_book_count' in result.columns
        # Penguin appears 4 times
        assert result.loc[0, 'publisher_book_count'] == 4
        assert result.loc[2, 'publisher_book_count'] == 4
        assert result.loc[3, 'publisher_book_count'] == 4
        # HarperCollins appears twice
        assert result.loc[1, 'publisher_book_count'] == 2
        assert result.loc[6, 'publisher_book_count'] == 2

    def test_in_series_feature(self, sample_data):
        """Test that series membership flag is created"""
        result = fe_engineering(sample_data, encode_text_embeddings=False)

        assert 'in_series' in result.columns
        assert result.loc[0, 'in_series'] == 0
        assert result.loc[1, 'in_series'] == 1
        assert result.loc[2, 'in_series'] == 0
        assert result.loc[4, 'in_series'] == 1
        assert result.loc[6, 'in_series'] == 1

    def test_description_length_feature(self, sample_data):
        """Test that description length is calculated correctly"""
        result = fe_engineering(sample_data, encode_text_embeddings=False)

        assert 'description_length' in result.columns
        assert result.loc[0, 'description_length'] > 0
        assert result.loc[1, 'description_length'] > 0
        assert result.loc[2, 'description_length'] == 0  # None value
        assert result.loc[4, 'description_length'] == 5  # 'short'

    def test_description_word_count_feature(self, sample_data):
        """Test that description word count is calculated correctly"""
        result = fe_engineering(sample_data, encode_text_embeddings=False)

        assert 'description_word_count' in result.columns
        assert result.loc[0, 'description_word_count'] == 6
        assert result.loc[1, 'description_word_count'] == 7
        assert result.loc[2, 'description_word_count'] == 0  # None value
        assert result.loc[4, 'description_word_count'] == 1

    def test_boolean_columns_encoding(self, sample_data):
        """Test that boolean columns are encoded correctly"""
        result = fe_engineering(sample_data, encode_text_embeddings=False)

        assert 'has_awards_encoded' in result.columns
        assert 'is_major_publisher_encoded' in result.columns

        # Check encoding values
        assert result.loc[0, 'has_awards_encoded'] == 1
        assert result.loc[1, 'has_awards_encoded'] == 1
        assert result.loc[2, 'has_awards_encoded'] == 0
        assert result.loc[4, 'has_awards_encoded'] == 0

        assert result.loc[0, 'is_major_publisher_encoded'] == 1
        assert result.loc[1, 'is_major_publisher_encoded'] == 1
        assert result.loc[4, 'is_major_publisher_encoded'] == 0

    def test_explicit_bool_cols_parameter(self):
        """Test that explicit bool_cols parameter works"""
        df = pd.DataFrame({
            'book_id': [1, 2, 3],
            'custom_bool': [True, False, True],
            'another_flag': [False, True, False]
        })

        result = fe_engineering(
            df,
            bool_cols=['custom_bool', 'another_flag'],
            encode_text_embeddings=False
        )

        assert 'custom_bool_encoded' in result.columns
        assert 'another_flag_encoded' in result.columns
        assert result.loc[0, 'custom_bool_encoded'] == 1
        assert result.loc[1, 'custom_bool_encoded'] == 0

    def test_missing_columns_handled_gracefully(self):
        """Test that function works when columns are missing"""
        df = pd.DataFrame({
            'book_id': [1, 2, 3],
            'description_clean': ['Text 1', 'Text 2', 'Text 3']
        })

        result = fe_engineering(df, encode_text_embeddings=False)

        # Should have description features
        assert 'description_length' in result.columns
        assert 'description_word_count' in result.columns
        # Should not have other features
        assert 'genre_count' not in result.columns
        assert 'author_book_count' not in result.columns

    def test_text_embeddings_disabled_by_default(self, sample_data):
        """Test that text embeddings are not generated by default"""
        result = fe_engineering(sample_data)

        assert 'text_embedding' not in result.columns

    def test_text_embeddings_import_error(self, sample_data, monkeypatch):
        """
        Test that ImportError is raised when
        SentenceTransformer unavailable
        """
        from src.modeling import feature_engineering

        # Temporarily set ST_MODEL to None
        original_model = feature_engineering.ST_MODEL
        feature_engineering.ST_MODEL = None

        try:
            with pytest.raises(
                ImportError,
                match="SentenceTransformer not available"
            ):
                fe_engineering(sample_data, encode_text_embeddings=True)
        finally:
            feature_engineering.ST_MODEL = original_model

    def test_original_dataframe_not_modified(self, sample_data):
        """Test that original DataFrame is not modified"""
        original_columns = sample_data.columns.tolist()

        result = fe_engineering(sample_data, encode_text_embeddings=False)

        # Original should be unchanged
        assert sample_data.columns.tolist() == original_columns
        assert 'genre_count' not in sample_data.columns

        # Result should have new features
        assert 'genre_count' in result.columns

    def test_custom_column_names(self):
        """Test that custom column names are respected"""
        df = pd.DataFrame({
            'book_id': [1, 2],
            'my_text': ['Text 1', 'Text 2'],
            'my_genres': [['genre1'], ['genre2']],
            'my_author': ['Author A', 'Author B']
        })

        result = fe_engineering(
            df,
            text_col='my_text',
            genres_col='my_genres',
            author_col='my_author',
            encode_text_embeddings=False
        )

        assert 'description_length' in result.columns
        assert 'genre_count' in result.columns
        assert 'author_book_count' in result.columns

    def test_all_features_created_together(self, sample_data):
        """Test that all features are created in a single call"""
        result = fe_engineering(sample_data, encode_text_embeddings=False)

        expected_features = [
            'genre_count',
            'has_genres',
            'is_top_genre',
            'author_book_count',
            'is_top_author',
            'publisher_book_count',
            'in_series',
            'description_length',
            'description_word_count',
            'has_awards_encoded',
            'is_major_publisher_encoded'
        ]

        for feature in expected_features:
            assert feature in result.columns

    def test_top_n_parameters(self):
        """
        Test that top_n_authors and top_n_genres
        parameters work correctly
        """
        df = pd.DataFrame({
            'author_clean': ['A', 'A', 'A', 'B', 'B', 'C'],
            'genres_clean': [
                ['g1'], ['g1'], ['g1'],
                ['g2'], ['g2'], ['g3']
            ]
        })

        # if top_n_authors=1, only 'A' should be top author
        result = fe_engineering(
            df,
            top_n_authors=1,
            top_n_genres=1,
            encode_text_embeddings=False
        )

        assert result.loc[0, 'is_top_author'] == 1
        assert result.loc[3, 'is_top_author'] == 0
        assert result.loc[5, 'is_top_author'] == 0

        assert result.loc[0, 'is_top_genre'] == 1
        assert result.loc[3, 'is_top_genre'] == 0

    def test_empty_dataframe(self):
        """Test that function handles empty DataFrame"""
        df = pd.DataFrame()

        result = fe_engineering(df, encode_text_embeddings=False)

        assert len(result) == 0
        assert isinstance(result, pd.DataFrame)

    def test_genre_set_vs_list(self):
        """Test that function handles both sets and lists for genres"""
        df = pd.DataFrame({
            'genres_clean': [
                ['genre1', 'genre2'],
                {'genre1', 'genre3'},
                ['genre1']
            ]
        })

        result = fe_engineering(df, encode_text_embeddings=False)

        assert result.loc[0, 'genre_count'] == 2
        assert result.loc[1, 'genre_count'] == 2
        assert result.loc[2, 'genre_count'] == 1
