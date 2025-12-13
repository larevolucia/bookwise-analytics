""" Model Runner Page: Predict Engagement for a Selected Book """
import tempfile
import os
import json
import streamlit as st
import pandas as pd
import joblib
import requests
import altair as alt
from sklearn.exceptions import NotFittedError
from src.analysis.analysis import (
    get_top_predicted_books
)

HF_BASE_URL = "https://huggingface.co/datasets/revolucia/"
DATASET_PATH = "bookwise-analytics-ml/resolve/main/modeling_data/"
MODEL_PATH = "popularity-score-model/"


def _load_eval_predictions_with_actuals():
    """Load eval predictions with actuals from the local notebook output."""
    try:
        df = pd.read_csv("outputs/model_plots/model_eval_predictions.csv")
        return df, "actual_score", "predicted_score"
    except (
        FileNotFoundError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        OSError
    ):
        return None, None, None


def _render_model_metrics():
    """Display model performance metrics (R² and RMSE) from saved JSON file."""
    metrics_path = os.path.join(
        "outputs",
        "models",
        "extratree",
        "3",
        "metrics.json"
    )

    try:
        with open(metrics_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        r2 = metrics.get("r2", 0)
        rmse = metrics.get("rmse", 0)
        st.caption(f"R²: {r2:.3f} | RMSE: {rmse:.3f}")
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        st.error("Could not load model metrics.")


def _render_residuals_plot(eval_df):
    """Render residuals vs predicted plot."""
    eval_df_copy = eval_df.copy()
    eval_df_copy['residual'] = (
        eval_df_copy['actual_score'] - eval_df_copy['predicted_score']
    )
    scatter = alt.Chart(eval_df_copy).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X('predicted_score', title='Predicted engagement score'),
        y=alt.Y('residual', title='Residual (Actual - Predicted)'),
        tooltip=['predicted_score', 'residual']
    )
    zero_line = pd.DataFrame({'y': [0]})
    ref_line = alt.Chart(zero_line).mark_rule(color='firebrick').encode(y='y')
    chart = (scatter + ref_line).properties(height=360)
    return chart


def _render_model_performance(
    model,
    features_df,
    actual_col="popularity_score",
    predicted_col=None
):
    """Plot predicted vs actual engagement scores
    with metrics
    returns bool success."""
    if features_df is None:
        return False
    if actual_col not in features_df.columns:
        return False

    eval_df = features_df.dropna(subset=[actual_col]).copy()
    if eval_df.empty:
        return False

    try:
        if predicted_col and predicted_col in eval_df.columns:
            eval_df["predicted_score"] = eval_df[predicted_col]
        else:
            if model is None:
                return False
            features_for_pred = eval_df.drop(
                columns=[actual_col, "title_clean", "goodreads_id_clean"],
                errors="ignore"
            )
            eval_df["predicted_score"] = model.predict(features_for_pred)

        eval_df["actual_score"] = eval_df[actual_col]

        diag_min = (
            float(eval_df[["actual_score", "predicted_score"]].min().min())
        )
        diag_max = (
            float(eval_df[["actual_score", "predicted_score"]].max().max())
        )
        scatter = alt.Chart(eval_df).mark_circle(size=60, opacity=0.6).encode(
            x=alt.X("actual_score", title="Actual engagement score"),
            y=alt.Y("predicted_score", title="Predicted engagement score"),
            tooltip=["actual_score", "predicted_score"]
        )
        diagonal = alt.Chart(
            pd.DataFrame({
                "actual_score": [diag_min, diag_max],
                "predicted_score": [diag_min, diag_max]
            })
        ).mark_line(color="firebrick").encode(
            x="actual_score",
            y="predicted_score"
        )
        st.altair_chart(
            (scatter + diagonal).properties(height=360),
            use_container_width=True
        )
    except (ValueError, TypeError, AttributeError, NotFittedError) as exc:
        st.error(f"Could not plot predicted vs actual: {exc}")
        return False

    return True


def _safe_hf_csv(
    filename: str,
    caption: str = "",
    subfolder: str = "modeling_data"
):
    """
    Load CSV directly from Hugging Face Hub from 'modeling_data' subfolder.
    """
    csv_url = (
        f"{HF_BASE_URL}bookwise-analytics-ml/resolve/main/"
        f"{subfolder}/{filename}"
    )
    try:
        df = pd.read_csv(csv_url)
        if caption:
            st.caption(caption)
        return df
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError):
        st.error(
            f"Could not load: {filename} "
            f"from Hugging Face Hub ({subfolder})."
        )
        return None


def _load_hf_model_pkl(filename: str, subfolder: str = "modeling_data"):
    """
    Download and load a pickle model from Hugging Face Hub (model repo).
    """
    # Use the model repo URL, not datasets
    model_url = (
        f"https://huggingface.co/"
        "revolucia/popularity-score-model/resolve/main/"
        f"{subfolder}/{filename}"
    )
    try:
        with requests.get(model_url, stream=True, timeout=30) as r:
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pkl"
            ) as tmp:
                for chunk in r.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                tmp_path = tmp.name
        model = joblib.load(tmp_path)
        return model
    except (
        requests.RequestException,
        joblib.externals.loky.process_executor.TerminatedWorkerError
    ) as exc:
        st.error(f"Could not load model from Hugging Face: {exc}")
        return None


def page_model_runner_body():
    """
    Run the trained model to help select new titles for the internal catalog.
    """
    st.write("## Catalog Title Selector")

    st.info(
        "**Business Requirement 2: High-Engagement Prediction**\n\n"
        "This page displays the books from the supply catalog with the "
        "highest predicted engagement scores. Use the search bar to look up "
        "the predicted score for any specific book.\n\n"
        "**Success Indicators:**\n"
        "* Model RMSE < 1.0 or R² > 0.7."
    )

    st.write("---")

    st.markdown(
        """
        **How to interpret the predicted engagement score:**

        The score is a composite metric based on the sum of z-scores
        (standardized values) for reviews, number of ratings, and average
        rating.

        - **A score of 0** means the book is average compared to the dataset.
        - **Positive values** indicate above-average predicted engagement
        (more reviews, higher rating, or more ratings than typical).
        - **Negative values** indicate below-average predicted engagement.
        """
    )

    st.write("---")

    model_filename = "et_model.pkl"

    st.write(
        "Load model and data from Hugging Face Hub "
        "to support catalog selection."
    )
    # Load data from Hugging Face Hub
    catalog_df = _safe_hf_csv(
        "supply_catalog_analysis.csv",
        caption="Supply Catalog (Hugging Face)"
    )
    features_df = _safe_hf_csv(
        "supply_catalog_final_features.csv",
        caption="Feature Dataset (Hugging Face)"
    )
    if catalog_df is None or features_df is None:
        st.error("Could not load required datasets from Hugging Face.")
        return

    # Load model from Hugging Face (from modeling_data folder)
    model = _load_hf_model_pkl(model_filename)
    if model is None:
        return

    st.write("## Model Performance Context")
    plotted = _render_model_performance(model, features_df)
    if not plotted:
        eval_df, act_col, pred_col = _load_eval_predictions_with_actuals()
        if eval_df is None:
            st.info(
                "No evaluation file with actuals found. "
                "Save model_eval_predictions.csv "
                "(actual_score, predicted_score) to outputs/model_plots/ "
                "to enable this chart."
            )
        else:
            st.write("Validation Predictions")
            _render_model_metrics()

            # eesponsive layout: side-by-side on desktop, stacked on mobile
            cols = st.columns(2)
            with cols[0]:
                _render_model_performance(
                    model=None,
                    features_df=eval_df,
                    actual_col=act_col,
                    predicted_col=pred_col
                )
                st.caption(
                    "A strong linear relationship between actual and "
                    "predicted scores indicates the model generalizes well. "
                    "Most points clustered near the diagonal line show the "
                    "model predicts book popularity accurately for the "
                    "majority of cases, with only a few outliers."
                )
            with cols[1]:
                residuals_chart = _render_residuals_plot(eval_df)
                st.altair_chart(residuals_chart, use_container_width=True)
                st.caption(
                    "Most residuals centered around zero indicate unbiased "
                    "predictions overall. Random scatter suggests good model "
                    "fit with no systematic errors. Consistent residual "
                    "patterns across validation and test sets confirm that "
                    "error characteristics generalize well to unseen data."
                )

    st.write("## Select from Supply Catalog")
    st.write(
        "Review top predicted books and select promising new titles for the "
        "internal catalog."
    )
    try:
        features_path = (
            f"{HF_BASE_URL}{DATASET_PATH}supply_catalog_final_features.csv"
        )
        catalog_path = (
            f"{HF_BASE_URL}{DATASET_PATH}supply_catalog_analysis.csv"
        )
        top_books = get_top_predicted_books(
            model=model,
            features_path=features_path,
            catalog_path=catalog_path,
            top_n=15
        )
        st.write("### Top 15 Recommended Titles")
        st.dataframe(top_books)

    except (FileNotFoundError, pd.errors.ParserError, OSError) as e:
        st.error(f"Could not compute top predicted books: {e}")

    st.write("### Predict Engagement for a Selected Title")
    st.write(
        "Choose a title from the entire supply catalog to see its predicted "
        "engagement score and decide if it fits the catalog."
    )
    book_titles = catalog_df["title_clean"].dropna().unique()
    selected_title = st.selectbox("Select a book title", book_titles)

    selected_row = (
        catalog_df[catalog_df["title_clean"] == selected_title].iloc[0]
    )
    selected_id = selected_row["goodreads_id_clean"]

    feature_row = features_df[features_df["goodreads_id_clean"] == selected_id]
    if feature_row.empty:
        st.error("No features found for the selected book.")
        return

    features = feature_row.drop(
        labels=[
            "popularity_score",
            "title_clean",
            "goodreads_id_clean"
        ],
        errors="ignore"
    )
    try:
        predicted_score = model.predict(features)[0]
        st.write(f"**Predicted Engagement Score:** {predicted_score:.2f}")
    except (ValueError, TypeError, AttributeError, NotFittedError) as exc:
        st.error(f"Prediction failed: {exc}")
