"""
Visualization functions for analyzing single catalog distributions.

This module provides plotting utilities for exploring individual
catalog characteristics and distributions.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_top_books(df, catalog_name, n=20):
    """
    Display top-rated books in a catalog.

    Parameters:
    -----------
    df : pd.DataFrame
        Catalog with title_clean and rating_clean columns
    catalog_name : str
        Name of the catalog for plot title
    n : int
        Number of top books to display

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    top_books = (
        df.sort_values("rating_clean", ascending=False)
          .head(n)[["title_clean", "rating_clean"]]
    )

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot horizontal bar chart
    ax.barh(range(len(top_books)), top_books["rating_clean"].values)

    # Set y-axis labels
    ax.set_yticks(range(len(top_books)))
    ax.set_yticklabels(top_books["title_clean"].values)

    # Set x-axis limits to zoom into the data range
    min_rating = top_books["rating_clean"].min()
    max_rating = top_books["rating_clean"].max()
    padding = (max_rating - min_rating) * 0.1  # 10% padding
    ax.set_xlim(min_rating - padding, max_rating + padding)

    # Invert y-axis so highest rating is at top
    ax.invert_yaxis()

    ax.set_xlabel("Rating")
    ax.set_title(
        f"Top {n} Rated Books — {catalog_name}",
        fontweight='bold'
    )
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_top_genres(df, catalog_name, min_count=20):
    """
    Display top-rated genres with minimum book count threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        Catalog with genres_clean and rating_clean columns
    catalog_name : str
        Name of the catalog for plot title
    min_count : int
        Minimum number of books required per genre
    """
    g = (
        df.explode("genres_clean")
          .groupby("genres_clean")["rating_clean"]
          .agg(["mean", "count"])
    )

    g = g[g["count"] >= min_count].sort_values(
        "mean", ascending=False
    ).head(20)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    g["mean"].plot(kind="bar", ax=ax)

    # Set y-axis limits to zoom into the data range
    min_rating = g["mean"].min()
    max_rating = g["mean"].max()
    padding = (max_rating - min_rating) * 0.1  # 10% padding
    ax.set_ylim(min_rating - padding, max_rating + padding)

    ax.set_title(
        f"Top Rated Genres (min {min_count} books) — {catalog_name}",
        fontweight='bold'
    )
    ax.set_ylabel("Avg Rating")
    ax.set_xlabel("Genre")
    plt.xticks(rotation=75, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_top_authors(df, catalog_name, min_books=5):
    """
    Display top-rated authors with minimum book count threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        Catalog with author_clean and rating_clean columns
    catalog_name : str
        Name of the catalog for plot title
    min_books : int
        Minimum number of books required per author

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    a = (
        df.groupby("author_clean")["rating_clean"]
          .agg(["mean", "count"])
    )

    a = a[a["count"] >= min_books].sort_values(
        "mean", ascending=False
    ).head(20)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    a["mean"].plot(kind="bar", ax=ax)

    # Set y-axis limits to zoom into the data range
    min_rating = a["mean"].min()
    max_rating = a["mean"].max()
    padding = (max_rating - min_rating) * 0.1  # 10% padding
    ax.set_ylim(min_rating - padding, max_rating + padding)

    ax.set_title(
        f"Top Rated Authors (min {min_books} books) — {catalog_name}",
        fontweight='bold'
    )
    ax.set_ylabel("Avg Rating")
    ax.set_xlabel("Author")
    plt.xticks(rotation=75, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_top_publishers(df, catalog_name, min_books=5):
    """
    Display top-rated publishers with minimum book count threshold.

    Parameters:
    -----------
    df : pd.DataFrame
        Catalog with publisher_clean and rating_clean columns
    catalog_name : str
        Name of the catalog for plot title
    min_books : int
        Minimum number of books required per publisher

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    p = (
        df.groupby("publisher_clean")["rating_clean"]
          .agg(["mean", "count"])
    )

    p = p[p["count"] >= min_books].sort_values(
        "mean", ascending=False
    ).head(20)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    p["mean"].plot(kind="bar", ax=ax)

    # Set y-axis limits to zoom into the data range
    min_rating = p["mean"].min()
    max_rating = p["mean"].max()
    padding = (max_rating - min_rating) * 0.1  # 10% padding
    ax.set_ylim(min_rating - padding, max_rating + padding)

    ax.set_title(
        f"Top Rated Publishers (min {min_books} books) — {catalog_name}",
        fontweight='bold'
    )
    ax.set_ylabel("Avg Rating")
    ax.set_xlabel("Publisher")
    plt.xticks(rotation=75, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    return fig


def plot_awards_rating(df, catalog_name):
    """
    Compare average ratings for awarded vs non-awarded books.

    Parameters:
    -----------
    df : pd.DataFrame
        Catalog with has_awards and rating_clean columns
    catalog_name : str
        Name of the catalog for plot title
    """
    avg = df.groupby("has_awards")["rating_clean"].mean()

    avg.plot(kind="bar", figsize=(6, 4))
    plt.title(
        f"Avg Rating — Awarded vs Non-Awarded ({catalog_name})"
    )
    plt.ylabel("Avg Rating")
    plt.xticks(rotation=0)
    plt.show()


def plot_boolean_pie_chart(
    df,
    boolean_col,
    catalog_name='Catalog',
    true_label='True',
    false_label='False',
    title=None,
    colors=None
):
    """
    Display pie chart showing percentage distribution of a boolean feature.

    Parameters:
    -----------
    df : pd.DataFrame
        Catalog dataframe
    boolean_col : str
        Name of the boolean column (e.g., 'is_major_publisher', 'has_award')
    catalog_name : str, default='Catalog'
        Name of the catalog for plot title
    true_label : str, default='True'
        Label for True values (e.g., 'Major Publisher', 'Has Award')
    false_label : str, default='False'
        Label for False values (e.g., 'Indie Publisher', 'No Award')
    title : str, optional
        Custom plot title. If None, generates from boolean_col
    colors : list, optional
        Custom colors for [False, True]. If None, uses default palette

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object
    """
    # Count boolean values
    counts = df[boolean_col].value_counts()

    # Reindex to ensure both True and False are present
    counts = counts.reindex([False, True], fill_value=0)

    # Calculate percentages
    total = counts.sum()
    percentages = (counts / total * 100).round(1)

    # Prepare labels with counts and percentages
    labels = [
        f'{false_label}\n{counts[False]:,} ({percentages[False]:.1f}%)',
        f'{true_label}\n{counts[True]:,} ({percentages[True]:.1f}%)'
    ]

    # Set colors
    if colors is None:
        colors = ['#ff9999', '#66b3ff']  # Light red, light blue

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create pie chart
    _wedges, _texts, autotexts = ax.pie(
        counts.values,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0.05, 0.05),  # Slight separation
        shadow=True,
        textprops={'fontsize': 11}
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)

    # Set title
    if title is None:
        col_display = boolean_col.replace('_', ' ').title()
        title = f'{col_display} Distribution — {catalog_name}'

    ax.set_title(title, fontweight='bold', fontsize=14, pad=20)

    plt.tight_layout()
    return fig


def plot_genre_bubble_chart(df, catalog_name):
    """
    Bubble chart of avg rating vs genre book count,
    bubble size scaled by total rating volume.
    df must contain: genres_clean (exploded),
    rating_clean, numRatings_clean.
    """

    # Aggregate at genre level
    genre_stats = (
        df.groupby("genres_clean")
          .agg(
              avg_rating=("rating_clean", "mean"),
              book_count=("genres_clean", "count"),
              rating_volume=("numRatings_clean", "sum")
          ).reset_index()
    )

    # Remove genres with too few books (optional filter)
    genre_stats = genre_stats[genre_stats["book_count"] >= 5]

    # Scale bubble size
    genre_stats["bubble_size"] = np.sqrt(genre_stats["rating_volume"] + 1) * 3

    plt.figure(figsize=(14, 8))
    sns.scatterplot(
        data=genre_stats,
        x="book_count",
        y="avg_rating",
        size="bubble_size",
        hue="avg_rating",
        palette="viridis",
        legend=False,
        alpha=0.6,
    )

    plt.title(f"Genre Stability Bubble Chart — {catalog_name}")
    plt.xlabel("Number of Books in Genre")
    plt.ylabel("Average Rating")
    plt.grid(True, alpha=0.3)

    for _, row in genre_stats.nlargest(15, "avg_rating").iterrows():
        plt.text(
            row.book_count,
            row.avg_rating + 0.003,
            row.genres_clean,
            fontsize=9
        )

    plt.show()
