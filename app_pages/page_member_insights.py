"""User Cluster Segmentation Insights Streamlit Page."""
import io
import streamlit as st
import pandas as pd
import joblib
import requests
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def page_member_insights_body():
    """Streamlit page body for User Cluster Segmentation Insights."""
    st.write("## User Cluster Segmentation Insights")

    st.markdown("""
    This page presents the results of user segmentation using KMeans
    clustering on member profile features.
    We identified **two main user clusters** based on reading behavior and
    preferences:
    - **Cluster 0: Genre Specialists** Fewer ratings, higher average rating,
    prefer newer and longer books, less genre diversity.
    - **Cluster 1: Genre Explorers** More ratings, slightly lower average
    rating, prefer older and shorter books, higher genre diversity.

    These segments help inform marketing, personalization, and engagement
    strategies.
    """)

    # Load data
    data_url = (
        "https://huggingface.co/datasets/revolucia/"
        "bookwise-analytics-ml/resolve/main/modeling_data/"
        "user_profile_features.csv"
    )

    model_url = (
        "https://huggingface.co/revolucia/bookclub-cluster/"
        "resolve/main/modeling_data/kmeans_model.pkl"
    )

    # Load data
    try:
        user_profiles = pd.read_csv(data_url)
    except Exception as e:
        st.error(f"Failed to load user profile data: {e}")
        return

    # Download model file and load with joblib
    try:
        response = requests.get(model_url, timeout=10)
        response.raise_for_status()
        kmeans = joblib.load(io.BytesIO(response.content))
    except Exception as e:
        st.error(f"Failed to download or load the clustering model: {e}")
        return

    # Prepare features (must match clustering features)
    num_cols = [
        'pages_mean', 'num_genres', 'genre_diversity', 'genre_concentration',
        'top_genre_share', 'num_interactions'
    ]
    cat_cols = ['diversity_category']

    if 'user_id' in user_profiles.columns:
        user_profiles = user_profiles.drop(columns=['user_id'])

    if 'cluster' not in user_profiles.columns:
        try:
            # Preprocessing (as in notebook)
            user_profiles[num_cols] = (
                SimpleImputer(strategy='median').fit_transform(
                    user_profiles[num_cols]
                )
            )
            user_profiles = pd.get_dummies(
                user_profiles, columns=cat_cols, drop_first=True
            )
            user_profiles[num_cols] = StandardScaler().fit_transform(
                user_profiles[num_cols]
            )
            user_profiles['cluster'] = kmeans.predict(user_profiles[num_cols])
        except Exception as e:
            st.error(f"Error during preprocessing or clustering: {e}")
            return

    # Show cluster sizes
    st.subheader("Cluster Sizes")
    cluster_counts = user_profiles['cluster'].value_counts().sort_index()
    st.bar_chart(cluster_counts)

    # Show cluster profiles (only key features for explanation)
    st.subheader("Key Features by Cluster")
    explain_cols = [
        "rating_count_x",
        "rating_mean",
        "pub_year_mean",
        "pages_mean",
        "genre_diversity",
        "num_interactions"
    ]
    if all(col in user_profiles.columns for col in explain_cols):
        cluster_profile = user_profiles.groupby('cluster')[explain_cols].mean()
        st.dataframe(cluster_profile.style.format("{:.2f}"))
    else:
        st.warning("Some key columns for explanation are missing in the data.")

    st.info("""
    **Interpretation:**
    - **Genre Specialists** are more focused in their reading and rate
    fewer, newer, longer books.
    - **Genre Explorers** are more diverse in their reading and rate more,
    older, shorter books.
    """)

    st.write("---"
             )
    st.write("### Advice")

    st.success(
        "**Recommendation:** Understanding these user segments allows for "
        "tailored marketing and personalization strategies. "
        "**Genre Specialists** may respond well to  targeted "
        "recommendations of new releases in their preferred genres, "
        "while **Genre Explorers** may appreciate curated lists that "
        "highlight diverse genres and authors. "
        "Combining strategies may yield the best member engagement outcomes."
    )

    st.write("---")
