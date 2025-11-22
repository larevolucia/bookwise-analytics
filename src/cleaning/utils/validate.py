""" Validation utilities for data cleaning. """
import ast
import numpy as np


def validate_cleaning_results(df, fields, suffix='_clean'):
    """
    Validate cleaning results by checking completeness and data types.

    Args:
        df: DataFrame with cleaned columns
        fields: List of field names to validate
        suffix: Suffix of cleaned columns

    Returns:
        Dictionary with validation statistics
    """
    results = {}

    for field in fields:
        col = f'{field}{suffix}'
        if col not in df.columns:
            results[field] = {'status': 'missing', 'completeness': 0}
            continue

        total = len(df)
        non_null = df[col].notna().sum()
        completeness = (non_null / total * 100) if total > 0 else 0

        results[field] = {
            'status': 'present',
            'completeness': round(completeness, 2),
            'non_null_count': non_null,
            'null_count': total - non_null,
            'dtype': str(df[col].dtype)
        }

    return results


def log_cleaning_stats(df_before, df_after, field_name):
    """
    Log before/after statistics for a cleaning operation.

    Args:
        df_before: DataFrame before cleaning
        df_after: DataFrame after cleaning
        field_name: Name of the field that was cleaned
    """
    before_null = df_before[field_name].isna().sum()
    after_null = df_after[field_name].isna().sum()
    before_unique = df_before[field_name].nunique()
    after_unique = df_after[field_name].nunique()

    print(f"\n{field_name} cleaning summary:")
    print(f"  Null values: {before_null} → {after_null}"
          f" (Δ {before_null - after_null})")
    print(f"  Unique values: {before_unique} → {after_unique}")
    total = len(df_after)
    completeness = ((total - after_null) / total * 100) if total > 0 else 0
    print(f"  Completeness: {completeness:.2f}%")


def compare_datasets(df1, df2, key_field, compare_fields):
    """
    Compare cleaning results across two datasets.

    Args:
        df1: First DataFrame
        df2: Second DataFrame
        key_field: Field to join on
        compare_fields: List of fields to compare

    Returns:
        DataFrame showing differences
    """
    merged = df1[[key_field] + compare_fields].merge(
        df2[[key_field] + compare_fields],
        on=key_field,
        suffixes=('_df1', '_df2'),
        how='outer'
    )

    # Find rows with differences
    diff_mask = False
    for field in compare_fields:
        diff_mask |= (merged[f'{field}_df1'] != merged[f'{field}_df2'])

    return merged[diff_mask]


def safe_sum_ratings(row):
    """ Safely sum ratingsByStars and compare to numRatings."""
    val = row['ratingsByStars_clean']
    # Convert stringified lists into Python lists
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return np.nan

    # If it's a list, make sure elements are integers
    if isinstance(val, list):
        try:
            val = [int(v) for v in val]  # convert each element to int
            return sum(val) == int(row['numRatings'])
        except (ValueError, TypeError):
            return np.nan

    return np.nan

# https://dev.to/mstuttgart/using-literal-eval-for-string-to-object-conversion-in-python-46i
# https://www.educative.io/answers/what-is-astliteralevalnodeorstring-in-python
