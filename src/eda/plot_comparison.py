"""
Visualization functions for comparing two catalogs.

This module provides plotting utilities for comparing internal
and supply catalogs side-by-side.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_genre_comparison(internal_df, supply_df, top_n=30):
    """
    Compare genre distribution between internal
    and supply catalogs using percentages.

    Parameters:
    -----------
    internal_df : pd.DataFrame
        Internal catalog with 'genres_clean' column
    supply_df : pd.DataFrame
        Supply catalog with 'genres_clean' column
    top_n : int
        Number of top genres to display
    """
    # Explode and count genres (assuming they're already lists)
    internal_genres = (
        internal_df.explode('genres_clean')['genres_clean'].dropna()
    )
    supply_genres = supply_df.explode('genres_clean')['genres_clean'].dropna()

    # Get total counts for percentage calculation
    internal_total = len(internal_genres)
    supply_total = len(supply_genres)

    # Count occurrences and convert to percentages
    internal_counts = (
        internal_genres.value_counts().head(top_n) / internal_total * 100
    )
    supply_counts = (
        supply_genres.value_counts().head(top_n) / supply_total * 100
    )

    # Get union of top genres from both catalogs
    all_genres = sorted(set(internal_counts.index) | set(supply_counts.index))

    # Prepare data
    internal_pct = [internal_counts.get(g, 0) for g in all_genres]
    supply_pct = [supply_counts.get(g, 0) for g in all_genres]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(all_genres))
    width = 0.35

    ax.barh(
        [i - width/2 for i in x],
        internal_pct,
        width,
        label='Internal Catalog',
        alpha=0.8
    )
    ax.barh(
        [i + width/2 for i in x],
        supply_pct,
        width,
        label='Supply Catalog',
        alpha=0.8
    )

    ax.set_yticks(x)
    ax.set_yticklabels(all_genres)
    ax.set_xlabel('Percentage of Books (%)')
    ax.set_title(
        f'Genre Distribution Comparison (Top {top_n} Genres)',
        fontweight='bold'
    )
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_author_comparison(internal_df, supply_df, top_n=30):
    """
    Compare author distribution between internal
    and supply catalogs using percentages.

    Parameters:
    -----------
    internal_df : pd.DataFrame
        Internal catalog with 'author_clean' column
    supply_df : pd.DataFrame
        Supply catalog with 'author_clean' column
    top_n : int
        Number of top authors to display
    """
    # Count authors
    internal_authors = internal_df['author_string'].dropna()
    supply_authors = supply_df['author_string'].dropna()

    # Get total counts for percentage calculation
    internal_total = len(internal_authors)
    supply_total = len(supply_authors)

    # Count occurrences and convert to percentages
    internal_counts = (
        internal_authors.value_counts().head(top_n) / internal_total * 100
    )
    supply_counts = (
        supply_authors.value_counts().head(top_n) / supply_total * 100
    )

    # Get union of top authors from both catalogs
    all_authors = sorted(
        set(internal_counts.index) | set(supply_counts.index)
    )

    # Prepare data
    internal_pct = [internal_counts.get(a, 0) for a in all_authors]
    supply_pct = [supply_counts.get(a, 0) for a in all_authors]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    x = range(len(all_authors))
    width = 0.35

    ax.barh(
        [i - width/2 for i in x],
        internal_pct,
        width,
        label='Internal Catalog',
        alpha=0.8
    )
    ax.barh(
        [i + width/2 for i in x],
        supply_pct,
        width,
        label='Supply Catalog',
        alpha=0.8
    )

    ax.set_yticks(x)
    ax.set_yticklabels(all_authors)
    ax.set_xlabel('Percentage of Books (%)')
    ax.set_title(
        f'Author Distribution Comparison (Top {top_n} Authors)',
        fontweight='bold'
    )
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_rating_distributions(internal_df, supply_df):
    """
    Plot rating distributions for internal and supply catalogs.

    Parameters:
    -----------
    internal_df : pd.DataFrame
        Internal catalog with rating_clean column
    supply_df : pd.DataFrame
        Supply catalog with rating_clean column
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(
        internal_df["rating_clean"].dropna(),
        fill=True,
        label="Internal",
        alpha=0.5,
        ax=ax
    )
    sns.kdeplot(
        supply_df["rating_clean"].dropna(),
        fill=True,
        label="Supply",
        alpha=0.5,
        ax=ax
    )
    ax.set_title("Rating Distribution — Internal vs Supply", fontweight='bold')
    ax.set_xlabel("Rating")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_major_publisher_comparison(internal_df, supply_df):
    """
    Compare average ratings for major vs non-major publishers
    between internal and supply catalogs.

    Parameters:
    -----------
    internal_df : pd.DataFrame
        Internal catalog with 'is_major_publisher' and 'rating_clean'
    supply_df : pd.DataFrame
        Supply catalog with 'is_major_publisher' and 'rating_clean'
    """
    # Calculate average ratings by major publisher status
    internal_avg = (
        internal_df.groupby('is_major_publisher')['rating_clean']
        .mean()
        .reindex([False, True])
    )
    supply_avg = (
        supply_df.groupby('is_major_publisher')['rating_clean']
        .mean()
        .reindex([False, True])
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [0, 1]
    width = 0.35

    ax.bar(
        [i - width/2 for i in x],
        internal_avg.values,
        width,
        label='Internal Catalog',
        alpha=0.8
    )
    ax.bar(
        [i + width/2 for i in x],
        supply_avg.values,
        width,
        label='Supply Catalog',
        alpha=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(['Non-Major Publisher', 'Major Publisher'])
    ax.set_ylabel('Average Rating')
    ax.set_title(
        'Average Rating: Major vs Non-Major Publishers',
        fontweight='bold'
    )
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Set y-axis limits to zoom into the data range
    all_values = list(internal_avg.values) + list(supply_avg.values)
    min_rating = min(all_values)
    max_rating = max(all_values)
    padding = (max_rating - min_rating) * 0.15  # 15% padding
    ax.set_ylim(min_rating - padding, max_rating + padding)

    # Add value labels on bars
    for i, (int_val, sup_val) in enumerate(
        zip(internal_avg.values, supply_avg.values)
    ):
        ax.text(
            i - width/2, int_val + (max_rating - min_rating) * 0.01,
            f'{int_val:.2f}',
            ha='center', va='bottom', fontsize=9
        )
        ax.text(
            i + width/2, sup_val + (max_rating - min_rating) * 0.01,
            f'{sup_val:.2f}',
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    return fig


def plot_award_comparison(internal_df, supply_df):
    """
    Compare average ratings for books with awards vs without awards
    between internal and supply catalogs.

    Parameters:
    -----------
    internal_df : pd.DataFrame
        Internal catalog with 'has_award' and 'rating_clean'
    supply_df : pd.DataFrame
        Supply catalog with 'has_award' and 'rating_clean'
    """
    # Calculate average ratings by award status
    internal_avg = (
        internal_df.groupby('has_award')['rating_clean']
        .mean()
        .reindex([False, True])
    )
    supply_avg = (
        supply_df.groupby('has_award')['rating_clean']
        .mean()
        .reindex([False, True])
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = [0, 1]
    width = 0.35

    ax.bar(
        [i - width/2 for i in x],
        internal_avg.values,
        width,
        label='Internal Catalog',
        alpha=0.8
    )
    ax.bar(
        [i + width/2 for i in x],
        supply_avg.values,
        width,
        label='Supply Catalog',
        alpha=0.8
    )

    ax.set_xticks(x)
    ax.set_xticklabels(['No Award', 'Has Award'])
    ax.set_ylabel('Average Rating')
    ax.set_title(
        'Average Rating: Award Status Comparison',
        fontweight='bold'
    )
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Set y-axis limits to zoom into the data range
    all_values = list(internal_avg.values) + list(supply_avg.values)
    min_rating = min(all_values)
    max_rating = max(all_values)
    padding = (max_rating - min_rating) * 0.15  # 15% padding
    ax.set_ylim(min_rating - padding, max_rating + padding)

    # Add value labels on bars
    for i, (int_val, sup_val) in enumerate(
        zip(internal_avg.values, supply_avg.values)
    ):
        ax.text(
            i - width/2, int_val + (max_rating - min_rating) * 0.01,
            f'{int_val:.2f}',
            ha='center', va='bottom', fontsize=9
        )
        ax.text(
            i + width/2, sup_val + (max_rating - min_rating) * 0.01,
            f'{sup_val:.2f}',
            ha='center', va='bottom', fontsize=9
        )

    plt.tight_layout()
    return fig


def plot_publication_year_comparison(internal_df, supply_df):
    """
    Plot publication year distributions for internal and supply catalogs.

    Parameters:
    -----------
    internal_df : pd.DataFrame
        Internal catalog with publication_year_clean column
    supply_df : pd.DataFrame
        Supply catalog with publication_year_clean column

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot KDE for both catalogs
    sns.kdeplot(
        internal_df["publication_year_clean"].dropna(),
        fill=True,
        label="Internal Catalog",
        alpha=0.5,
        ax=ax
    )
    sns.kdeplot(
        supply_df["publication_year_clean"].dropna(),
        fill=True,
        label="Supply Catalog",
        alpha=0.5,
        ax=ax
    )

    ax.set_title(
        "Publication Year Distribution — Internal vs Supply",
        fontweight='bold',
        fontsize=14
    )
    ax.set_xlabel("Publication Year")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    return fig
