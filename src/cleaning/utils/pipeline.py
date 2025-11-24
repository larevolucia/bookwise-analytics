""" Cleaning pipeline utilities """
from .language import clean_language_field, load_language_mapping
from .dates import (
    parse_mixed_date,
    clean_date_string,
    format_date_iso
)
from .numeric import clean_pages_field, clean_price
from .categories import clean_genre_list, parse_list_field
from .identifiers import clean_isbn
from .text_cleaning import clean_title, clean_description
from .metadata_cleaning import (
    clean_and_split_authors,
    clean_series,
    clean_awards_list
)


def apply_cleaners_selectively(
    df,
    fields_to_clean=None,
    source_suffix='',
    target_suffix='_clean',
    inplace=False
):
    """
    Apply cleaning functions selectively to specified fields.

    Args:
        df: DataFrame containing columns to clean
        fields_to_clean: List of field names to clean.
                        Options: 'language', 'publication_date', 'pages',
                        'subjects', 'isbn', 'title', 'author', 'series',
                        'description', 'price', 'awards', 'genres'
                        If None, cleans all available fields.
        source_suffix: Suffix of source columns (e.g., '_openlib', '_bbe')
        target_suffix: Suffix for cleaned output columns (default: '_clean')
        inplace: If True, modifies df in place. Otherwise returns a copy.

    Returns:
        DataFrame with cleaned columns added

    Examples:
        # Clean only language and pages from API data
        df = apply_cleaners_selectively(
            df,
            fields_to_clean=['language', 'pages'],
            source_suffix='_openlib'
        )

        # Clean all available fields from raw BBE data
        df = apply_cleaners_selectively(
            df,
            fields_to_clean=None  # or omit this argument
        )
    """
    if not inplace:
        df = df.copy()

    # load language dict once for efficiency
    lang_dict = load_language_mapping()

    # Define field mappings:
    # field_name -> (source_col_pattern, cleaner_functions)
    field_cleaners = {
        'language': (
            f'language{source_suffix}',
            lambda col: df[col].apply(
                lambda x: clean_language_field(x, lang_dict)
            )
        ),
        'publication_date': (  # OpenLibrary uses 'publication_date'
            f'publication_date{source_suffix}',
            lambda col: df[col]
            .apply(clean_date_string)    # 1. Clean raw string
            .apply(parse_mixed_date)      # 2. Parse to datetime
            .apply(format_date_iso)       # 3. Format as ISO string
        ),
        'publishedDate': (  # Google Books uses 'publishedDate'
            f'publishedDate{source_suffix}',
            lambda col: df[col]
            .apply(clean_date_string)
            .apply(parse_mixed_date)
            .apply(format_date_iso)
        ),
        'pages': (
            f'pages{source_suffix}',
            lambda col: df[col].apply(clean_pages_field)
        ),
        'pageCount': (  # Google Books uses 'pageCount'
            f'pageCount{source_suffix}',
            lambda col: df[col].apply(clean_pages_field)
        ),
        'subjects': (  # OpenLibrary uses 'subjects'
            f'subjects{source_suffix}',
            lambda col: df[col].apply(clean_genre_list)
        ),
        'categories': (  # Google Books uses 'categories'
            f'categories{source_suffix}',
            lambda col: df[col].apply(clean_genre_list)
        ),
        'genres': (
            f'genres{source_suffix}',
            lambda col: df[col].apply(parse_list_field).apply(clean_genre_list)
        ),
        'isbn': (
            f'isbn{source_suffix}',
            lambda col: df[col].apply(clean_isbn)
        ),
        'title': (
            f'title{source_suffix}',
            lambda col: df[col].apply(clean_title)
        ),
        'author': (
            f'author{source_suffix}',
            lambda col: df[col].apply(clean_and_split_authors)
        ),
        'authors': (  # Alias for 'author'
            f'authors{source_suffix}',
            lambda col: df[col].apply(clean_and_split_authors)
        ),
        'series': (
            f'series{source_suffix}',
            lambda col: df[col].apply(clean_series)
        ),
        'description': (
            f'description{source_suffix}',
            lambda col: df[col].apply(clean_description)
        ),
        'price': (
            f'price{source_suffix}',
            lambda col: df[col].apply(clean_price)
        ),
        'awards': (
            f'awards{source_suffix}',
            lambda col: df[col].apply(
                parse_list_field
                ).apply(
                    clean_awards_list
                    )
        ),
    }

    # if no fields specified, clean all available fields
    if fields_to_clean is None:
        fields_to_clean = field_cleaners.keys()

    # apply cleaners for specified fields
    for field in fields_to_clean:
        if field not in field_cleaners:
            print(f"Warning: Unknown field '{field}', skipping.")
            continue

        source_col, cleaner_func = field_cleaners[field]
        target_col = f'{field}{target_suffix}'

        # only clean if source column exists
        if source_col in df.columns:
            df[target_col] = cleaner_func(source_col)
        else:
            # try without suffix (for raw data)
            alt_col = field
            if alt_col in df.columns:
                if field == 'language':
                    df[target_col] = df[alt_col].apply(
                        lambda x: clean_language_field(x, lang_dict)
                    )
                elif field in ['publication_date', 'publishedDate']:
                    df[target_col] = df[alt_col].apply(
                        clean_date_string
                        ).apply(
                            parse_mixed_date
                            ).apply(
                                format_date_iso
                                )
                elif field in ['pages', 'pageCount']:
                    df[target_col] = df[alt_col].apply(clean_pages_field)
                elif field in ['subjects', 'categories']:
                    df[target_col] = df[alt_col].apply(clean_genre_list)
                elif field == 'genres':
                    df[target_col] = df[alt_col].apply(
                        parse_list_field
                        ).apply(
                            clean_genre_list
                            )
                elif field == 'isbn':
                    df[target_col] = df[alt_col].apply(clean_isbn)
                elif field == 'title':
                    df[target_col] = df[alt_col].apply(clean_title)
                elif field in ['author', 'authors']:
                    df[target_col] = df[alt_col].apply(clean_and_split_authors)
                elif field == 'series':
                    df[target_col] = df[alt_col].apply(clean_series)
                elif field == 'description':
                    df[target_col] = df[alt_col].apply(clean_description)
                elif field == 'price':
                    df[target_col] = df[alt_col].apply(clean_price)
                elif field == 'awards':
                    df[target_col] = df[alt_col].apply(
                        parse_list_field
                        ).apply(
                            clean_awards_list
                            )

    return df
