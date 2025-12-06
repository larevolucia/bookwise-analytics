"""
This module defines the modeling pipeline for preprocessing and model training.
"""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import SelectFromModel


# custom transformers
class DropColumns(BaseEstimator, TransformerMixin):
    """ Transformer to drop specified columns from the dataset.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        """ Fit method"""
        return self

    def transform(self, X):
        """ Transform method to drop columns """
        return X.drop(columns=self.columns)


class FillNAWithValue(BaseEstimator, TransformerMixin):
    """ Transformer to fill NA values in specified columns with a given value.
    """
    def __init__(self, columns, value):
        self.columns = columns
        self.value = value

    def fit(self, X, y=None):
        """Fit method"""
        return self

    def transform(self, X):
        """Transform method to fill NA values"""
        X = X.copy()
        for col in self.columns:
            X[col] = X[col].fillna(self.value)
        return X


class AddMissingFlags(BaseEstimator, TransformerMixin):
    """ Transformer to add missing value flags for specified columns.
    """
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        """Fit method"""
        return self

    def transform(self, X):
        """Transform method to add missing value flags"""
        X = X.copy()
        for col in self.columns:
            X[f'{col}_was_missing'] = X[col].isnull().astype(int)
        return X


# pipeline construction function
def get_preprocessing_pipeline():
    """Constructs the preprocessing pipeline for modeling."""
    num_cols = [
        'external_likedpct', 'pages_log', 'external_score_log',
        'external_votes_log', 'external_numratings_log'
    ]
    cat_cols = ['publication_year', 'publication_decade']
    drop_cols = [
        'external_price_log',
        'language_final',
        'publication_date_final',
    ]
    fill_minus1_cols = ['external_rating', 'external_popularity_score']
    flag_cols = fill_minus1_cols + num_cols

    preprocessing_pipeline = Pipeline([
        ('drop_cols', DropColumns(drop_cols)),
        ('fill_minus1', FillNAWithValue(fill_minus1_cols, -1)),
        ('add_flags', AddMissingFlags(flag_cols)),
        ('col_transform', ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), num_cols),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_cols)
        ], remainder='passthrough')),
    ])
    return preprocessing_pipeline


def PipelineOptimization(model):
    """
    Constructs the full modeling pipeline
    with preprocessing and feature selection.
    """
    pipeline_base = Pipeline([
        ('preprocessing', get_preprocessing_pipeline()),
        ('feat_selection', SelectFromModel(model)),
        ('model', model),
    ])
    return pipeline_base
