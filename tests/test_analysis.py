""" Additional tests for analysis functions. """
import pandas as pd
import numpy as np
from src.analysis import analysis


class DummyModel:
    """A simple dummy model for testing purposes."""
    def predict(self, X):
        """Return a constant or sequence for testing"""
        return np.ones(len(X))


def test_calculate_genre_entropy():
    """
    Test genre entropy calculation.

    Verifies that `calculate_genre_entropy`
    returns a float when given a list of genres.
    This checks the function's output type, not its value.
    """
    genres = ['Fiction', 'Fiction', 'Nonfiction']
    entropy = analysis.calculate_genre_entropy(genres)
    assert isinstance(entropy, float)


def test_get_book_predicted_score_found():
    """
    Test getting predicted score for a known book.

    Checks that `get_book_predicted_score`
    returns the correct predicted score (`1.0`)
    for a book that exists in the DataFrame.
    It uses a dummy model that always predicts `1`.
    """
    model = DummyModel()
    df = pd.DataFrame({
        'goodreads_id_clean': ['abc'],
        'popularity_score': [1],
        'title_clean': ['Book'],
    })
    score = (
        analysis.get_book_predicted_score(model, df, 'abc')
    )
    assert score == 1.0


def test_get_book_predicted_score_not_found():
    """
    Test getting predicted score for an unknown book.

    Ensures that `get_book_predicted_score`
    returns `None` when the requested book ID
    is not found in the DataFrame.
    """
    model = DummyModel()
    df = pd.DataFrame({'goodreads_id_clean': ['abc']})
    score = analysis.get_book_predicted_score(model, df, 'xyz')
    assert score is None


def test_simulate_uplift():
    """
    Test uplift simulation.

    Confirms that `simulate_uplift`
    returns a float when given two DataFrames with
    `predicted_score` columns.
    It does not check the value, only the type.
    """
    editorial = pd.DataFrame({'predicted_score': [1, 2, 3]})
    model = pd.DataFrame({'predicted_score': [2, 3, 4]})
    uplift = analysis.simulate_uplift(editorial, model)
    assert isinstance(uplift, float)
