"""Tests for the modeling pipeline and related utilities.

This module contains tests that verify the behavior of custom transformers,
dummy models, and pipeline utilities used in the modeling process.

References:
https://www.fuzzylabs.ai/blog-post/the-art-of-testing-machine-learning-pipelines
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from src.modeling import modeling_pipeline


class DummyModel(BaseEstimator, ClassifierMixin):
    """
    A simple dummy model for testing pipelines.

    This model always predicts a constant value and provides a coefficient
    array for compatibility with feature selection tools.
    """

    def __init__(self, constant=0):
        """
        Initialize the dummy model.

        Args:
            constant (int, optional): The constant value to predict.
            Defaults to 0.
        """
        self.constant = constant

    def fit(self, X, y=None):
        """
        Pretend to fit the model to the data.

        This method does not actually learn from the data, but sets up
        attributes required by scikit-learn utilities.

        Args:
            X (pd.DataFrame): Input features.
            y (array-like, optional): Target values (unused).

        Returns:
            DummyModel: The fitted model.
        """
        self.feature_names_in_ = getattr(X, 'columns', None)
        self.coef_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        """
        Predict a constant value for each sample.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Array of constant predictions.
        """
        return np.full(shape=(X.shape[0],), fill_value=self.constant)

    def predict_proba(self, X):
        """
        Predict constant probabilities for each class.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            np.ndarray: Array of probabilities.
        """
        # Create a 2D array where each row is
        # [probability of class 0, probability of class 1],
        # repeated for every sample in X.
        # The probabilities are based on the constant value.
        return np.tile([1 - self.constant, self.constant], (X.shape[0], 1))


def test_drop_columns():
    """
    Test that DropColumns removes specified columns from a DataFrame.

    This test checks that the custom transformer correctly drops columns,
    which is important for cleaning up data before modeling.
    """
    df = pd.DataFrame({'a': [1], 'b': [2]})
    dropper = modeling_pipeline.DropColumns(['a'])
    result = dropper.transform(df)
    assert 'a' not in result.columns


def test_fill_na_with_value():
    """
    Test that FillNAWithValue fills missing values in a DataFrame.

    This test checks that missing (None)
    values are replaced with a specified value,
    which is a common data cleaning step before modeling.
    """
    df = pd.DataFrame({'a': [None, 2]})
    filler = modeling_pipeline.FillNAWithValue(['a'], 0)
    result = filler.transform(df)
    assert result['a'].iloc[0] == 0


def test_add_missing_flags():
    """
    Test that AddMissingFlags adds a flag column for missing values.

    This test checks that a new column is created to indicate which values
    were missing, which can be useful for some models.
    """
    df = pd.DataFrame({'a': [None, 2]})
    flagger = modeling_pipeline.AddMissingFlags(['a'])
    result = flagger.transform(df)
    assert 'a_was_missing' in result.columns


def test_get_preprocessing_pipeline_runs():
    """
    Test that the preprocessing pipeline runs without errors.

    This test checks that the pipeline can
    process a DataFrame with the expected
    columns and missing values,
    which is necessary for model training.
    """
    df = pd.DataFrame({
        'external_likedpct': [1.0],
        'pages_log': [1.0],
        'external_score_log': [1.0],
        'external_votes_log': [1.0],
        'external_numratings_log': [1.0],
        'publication_year': [2020],
        'publication_decade': [2020],
        'external_price_log': [1.0],
        'language_final': ['en'],
        'publication_date_final': ['2020-01-01'],
        'external_rating': [None],
        'external_popularity_score': [None],
    })
    pipe = modeling_pipeline.get_preprocessing_pipeline()
    pipe.fit_transform(df)


def test_pipeline_optimization_runs():
    """
    Test that PipelineOptimization can fit a dummy model.

    This test checks that the custom pipeline optimization utility can run
    with a dummy model and sample data, ensuring all steps work together.
    """
    df = pd.DataFrame({
        'external_likedpct': [1.0],
        'pages_log': [1.0],
        'external_score_log': [1.0],
        'external_votes_log': [1.0],
        'external_numratings_log': [1.0],
        'publication_year': [2020],
        'publication_decade': [2020],
        'external_price_log': [1.0],
        'language_final': ['en'],
        'publication_date_final': ['2020-01-01'],
        'external_rating': [None],
        'external_popularity_score': [None],
    })
    y_target = [1]
    model = DummyModel()
    pipe = modeling_pipeline.PipelineOptimization(model)
    pipe.fit(df, y_target)


def test_pipeline_with_feature_selection_and_dummy_model():
    """
    Test a pipeline with feature selection and a dummy model.

    This test loads a list of features, creates random data,
    and checks that the pipeline can fit the data without errors.
    """
    final_features = pd.read_csv(
        'outputs/datasets/modeling/1/final_features.csv'
    ).columns.tolist()

    dummy_df = pd.DataFrame(
        np.random.rand(10, len(final_features)),
        columns=final_features
    )
    dummy_y = np.random.randint(0, 2, size=10)

    dummy_pipe = Pipeline([
        ('feature_selection', SelectFromModel(DummyModel())),
        ('model', DummyModel())
    ])

    dummy_pipe.fit(dummy_df, dummy_y)
