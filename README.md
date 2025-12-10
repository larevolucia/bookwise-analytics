# Bookwise Analytics: Predictive Book Recommendation System

Live App: [Streamlit Dashboard](https://bookwise-analytics-1d891f772a24.herokuapp.com/)

## 1. Project Overview

**Purpose:**  
Bookwise Analytics is a data-driven recommendation system for a subscription-based book club. The goal is to optimize book selection and user engagement using Machine Learning (ML), replacing intuition-based curation with predictive analytics. The project delivers a Streamlit dashboard for stakeholders to explore insights, model outputs, and diversity metrics.

**Target Audience:**  
Business stakeholders, data practitioners, and editorial teams seeking to maximize user satisfaction and retention in a book subscription service.

---

## 2. Business Understanding (CRISP-DM: Business Understanding)

### 2.1. Problem Statement

Despite a stable subscriber base, engagement and credit redemption rates are declining due to poor book-member matches. The business needs to identify drivers of engagement and predict which books will maximize satisfaction and retention.

### 2.2. Business Objectives

- **Identify** book and features linked to higher engagement.
- **Predict** high-engagement titles using historical data.
- **Simulate** retention uplift from algorithmic recommendations.
- **Safeguard** genre diversity and fairness in recommendations.

### 2.3. Stakeholder Benefits

- **Users:** Receive better-matched book recommendations, increasing engagement.
- **Business:** Reduces churn, improves catalog utilization, and supports scalable editorial processes.
- **Editorial:** Focuses curation on high-impact and diverse titles.

---

## 3. Business Requirements & Mapping (LO2, LO3)

| ID   | Requirement                                                                 | Success Indicator                        | Dataset(s)         | Linked Dashboard Page      |
|------|-----------------------------------------------------------------------------|------------------------------------------|--------------------|---------------------------|
| BR-1 | Identify features correlated with engagement                                | Correlation ≥ 0.4                        | BBE                | Analytics Explorer        |
| BR-2 | Predict high-engagement titles                                              | Model RMSE < 1.0 or R² > 0.7             | BBE, Goodbooks     | Model Runner              |
| BR-3 | Estimate retention uplift from recommendations                              | Simulated uplift ≥ 10%                   | BBE, Goodbooks     | Recommendation Comparison |
| BR-4 | Maintain diversity/fairness in recommendations                             | Shannon Entropy ≥ editorial baseline     | BBE, Goodbooks     | Diversity Metrics         |

---

## 3a. User Stories

**[High Engagement Titles](https://github.com/larevolucia/bookwise-analytics/issues/21)**  
As an editorial team member, I want to see which books are predicted to have high engagement so I can focus curation efforts.

**[Engagement Uplift Prediction](https://github.com/larevolucia/bookwise-analytics/issues/18)**  
As a business stakeholder, I want to compare editorial vs. model-driven recommendations to understand uplift, so I can make informed decisions.

**[Feature Importance for Engagement](https://github.com/larevolucia/bookwise-analytics/issues/16)**  
As a business stakeholder, I want to understand which book features drive engagement so I can optimize catalog selection.

**[Genre Fairness ](https://github.com/larevolucia/bookwise-analytics/issues/19)  **  
As a stakeholder, I want to ensure recommendations maintain genre diversity and fairness, so I don't alienate any user segments.

**[Summary Dashboard](https://github.com/larevolucia/bookwise-analytics/issues/20)**  
As a stakeholder, I want an executive summary page showing KPIs and project overview, so I can quickly assess performance.

**[Title Acquisition](https://github.com/larevolucia/bookwise-analytics/issues/22)**  
As a user, I want to search for any book and see its predicted engagement score, so I can guide title acquisition decisions.

---

## 3b. User Stories Mapped to ML & Visualization Tasks

| User Story                        | ML Task / Visualization                        | Actions Required                                                                                   |
|------------------------------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------|
|  [High Engagement Titles](https://github.com/larevolucia/bookwise-analytics/issues/21)          | Engagement prediction, leaderboard            | Model scoring, leaderboard table                                              |
| [Engagement Uplift Prediction](https://github.com/larevolucia/bookwise-analytics/issues/18)       | Editorial vs. model uplift metric           | Display sets, calculate predicition for each set, calculate uplift                          |
| [Feature Importance for Engagement](https://github.com/larevolucia/bookwise-analytics/issues/16)  | Feature importance analysis                   | Train model, extract importances, visualize, actionable insights                         |
| [Genre Fairness ](https://github.com/larevolucia/bookwise-analytics/issues/19)                    | Genre diversity/fairness metrics              | Compute shares, entropy, visualize                                            |
| [Summary Dashboard](https://github.com/larevolucia/bookwise-analytics/issues/20)                  | Executive KPIs dashboard                      | Aggregate KPIs, overview, navigation                                             |
| [Title Acquisition](https://github.com/larevolucia/bookwise-analytics/issues/22)                  | Search + engagement prediction                | Search bar, engagement prediction          |

---

## 3c. Business Requirements & Mapping 
| ID   | Requirement                                                                 | Success Indicator                        | Dataset(s)         | Linked Dashboard Page      |
|------|-----------------------------------------------------------------------------|------------------------------------------|--------------------|---------------------------|
| BR-1 | Identify features correlated with engagement                                | Correlation ≥ 0.4                        | BBE                | Analytics Explorer        |
| BR-2 | Predict high-engagement titles                                              | Model RMSE < 1.0 or R² > 0.7             | BBE, Goodbooks     | Model Runner              |
| BR-3 | Estimate retention uplift from recommendations                              | Simulated uplift ≥ 10%                   | BBE, Goodbooks     | Recommendation Comparison |
| BR-4 | Maintain diversity/fairness in recommendations                             | Shannon Entropy ≥ editorial baseline     | BBE, Goodbooks     | Diversity Metrics         |

---

## Stretch Goal: User Clustering & Segmentation

As an additional feature, this project implements **user clustering** using KMeans to segment members based on their reading behavior and preferences. This segmentation helps identify distinct user groups ( "Genre Specialists" vs. "Genre Explorers") and supports more targeted marketing and personalization strategies.

- See `notebooks/07_Customer_Cluster.ipynb` for the clustering workflow and analysis.
- Explore the "Member Insights" page in the dashboard for cluster profiles.

*Note: Clustering is provided as a stretch goal to demonstrate unsupervised learning and user segmentation beyond the core recommendation system.*

---

## 4. Hypotheses & Validation 

| ID   | Hypothesis                                                                 | Validation Method                        | Outcome/Conclusion |
|------|---------------------------------------------------------------------------|------------------------------------------|--------------------|
| H1   | Books with high external ratings have higher engagement        | Correlation, regression                  | Confirmed: r > 0.4 |
| H2   | Historical rating/review patterns predict engagement with ~80% accuracy    | Regression, collaborative filtering      | Confirmed: RMSE < 1.0, R² > 0.7|
| H3   | Recent publications yield higher satisfaction                             | Feature importance, correlation          | Partially confirmed |
| H4   | Algorithmic selection increases engagement by ≥10% over editorial/random   | Uplift simulation                        | Confirmed: ≥10% uplift |

*See dashboard and notebooks for statistical evidence and plots supporting these conclusions.*

---

## 5. Datasets

| Dataset             | Source & Link                                                                 | Purpose                        |
|---------------------|------------------------------------------------------------------------------|--------------------------------|
| Best Books Ever     | [GitHub](https://github.com/scostap/goodreads_bbe_dataset)                   | Catalog metadata, ratings      |
| Goodbooks-10k       | [GitHub](https://github.com/zygmuntz/goodbooks-10k)                          | User behavior, ratings         |
| Overlap (BBE ∩ GB)  | Derived                                                                       | Metadata linking               |
| Open Library API    | [API](https://openlibrary.org/developers/api)                                | Metadata enrichment            |
| Google Books API    | [API](https://developers.google.com/books)                                   | Metadata enrichment            |

---

## 6. Data Collection & Preparation

- **Data Collection:** Jupyter notebooks fetch and audit datasets, including API enrichment.
- **Data Cleaning:** Handle missing values, standardize fields, and merge datasets.
- **Feature Engineering:** Create popularity, metadata scores and log-transform skewed features.
- **Data Preparation:** Documented in `/notebooks/` (see structure below).

---

## 7. Analytical & ML Tasks

| Requirement | Task                                      | Notebook(s)                  | Outcome/Metric                |
|-------------|-------------------------------------------|------------------------------|-------------------------------|
| BR-1        | Correlation, feature importance           | 04_Exploratory_Data_Analysis | Key predictors identified     |
| BR-2        | Regression, hybrid recommender            | 06_Modeling                  | RMSE, R², MAE         |
| BR-3        | Uplift simulation                         | 06_Modeling, dashboard       | % uplift in engagement        |
| BR-4        | Genre entropy, diversity metrics          | 04_Exploratory_Data_Analysis | Entropy, genre coverage       |

---

## 8. ML Business Case

- **Aim:** Predict book engagement to optimize recommendations and retention.
- **Learning Method:** Regression (RandomForest, XGBoost) and clustering.
- **Success Metrics:** RMSE < 1.0, R² > 0.7, ≥10% uplift, high genre entropy.
- **Model Output:** Predicted engagement scores, top-K recommendations.
- **Relevance:** Directly supports business KPIs for engagement and retention.

---

## 9. Dashboard Design

**Pages:**

- **Executive Summary:** KPIs (RMSE, R², uplift), summary plots. *(BR-2, BR-3)*
- **Analytics Explorer:** Correlations, feature importance, genre diversity. *(BR-1, BR-4)*
- **Recommendation Comparison:** Model vs. editorial vs. random selections, uplift plots. *(BR-3)*
- **Model Runner:** Top 10 books by predicted engagement. *(BR-2)*
- **Diversity Metrics:** Genre entropy, coverage plots. *(BR-4)*

*Each page includes textual interpretation of plots and clear statements on model performance.*

---

## 10. Project Structure & Notebooks

```
/notebooks
├── 01_Data_Collection.ipynb
├── 02_Data_Cleaning.ipynb
├── 03_Data_Imputation.ipynb
├── 04_Exploratory_Data_Analysis.ipynb
├── 05_Feature_Engineering.ipynb
├── 06_Modeling.ipynb
```
- Each notebook starts with objectives, inputs, and outputs.
- Data preparation and feature engineering are clearly documented.

---

## 11. Deployment & Local Development

- **Streamlit app:** `streamlit run app.py`
- **Heroku deployment:** See instructions in this README.
- **Environment:** Python, pip, virtualenv, requirements.txt, Procfile, setup.sh maintained.
- **Version Control:** All code managed in GitHub with clear commit history.

---

## 12. Model Evaluation & Business Impact

### Summary

The Bookwise Analytics recommendation engine was evaluated using multiple regression models, with the **ExtraTreesRegressor** selected for deployment due to its strong performance:

- **Test R²:** 0.81
- **Test RMSE:** 0.95
- **Test MAE:** 0.57

These results exceed the business success criteria (RMSE < 1.0 or R² > 0.7), confirming the model's reliability for predicting book engagement.

**Feature importance analysis** shows that external popularity signals (e.g., number of ratings, votes, composite popularity score) are the strongest predictors, with publication recency and select metadata features providing additional value.

### Recommendation Uplift & Diversity

On the dashboard's Recommendation Comparison page, the model-driven approach achieved:

- **Simulated Uplift:** **74.9%** (vs. editorial selection, far exceeding the 10% target)
- **Genre Entropy:** **2.32** (no loss in diversity compared to editorial curation)

This demonstrates that algorithmic recommendations can substantially increase predicted engagement while maintaining genre diversity and fairness.

### Business Impact

- **Higher Engagement:** Model recommendations are expected to significantly boost user engagement and credit redemption.
- **Diversity Maintained:** No reduction in genre diversity or fairness.
- **Actionable Insights:** Editorial teams gain transparency into drivers of engagement and can focus on high-impact titles.

---

**For full details and supporting analysis, see:**
- [`notebooks/06_Modeling.ipynb`](notebooks/06_Modeling.ipynb) — Model training, evaluation, and feature importance
- **Dashboard > Recommendation Comparison** — Simulated uplift and genre entropy metrics
- [`app_pages/page_recommendation_comparison.py`](app_pages/page_recommendation_comparison.py) — Implementation of comparison logic
---

## 13. References & Attribution

- All datasets are open-source and referenced.
- External code and resources are credited in code comments and this README.
- See [References](#references) section for full list.


---

## 14. Bug Fixes

| Ticket Title                                                                                                   | Error Description                                   | Resolution Description                      |
|----------------------------------------------------------------------------------------------------------------|-----------------------------------------------------|---------------------------------------------|
| [DtypeWarning when loading CSVs](https://github.com/larevolucia/bookwise-analytics/issues/10)                  | Mixed data types in columns causing warnings        | Specified dtypes when reading CSVs          |
| [Application error](https://github.com/larevolucia/bookwise-analytics/issues/11)                               | Heroku deployment failed due to conflicting packages| Refactored python `pyproject.toml`          |
| [ppscore dependency issue](https://github.com/larevolucia/bookwise-analytics/issues/12)                        | ppscore package causing installation errors         | Refactored python `pyproject.toml`          |
| [WEBP images from HF not displaying](https://github.com/larevolucia/bookwise-analytics/issues/14)              | Streamlit not rendering WEBP images from HF         | Removed empty space from image URL          |
| [Deployment error](https://github.com/larevolucia/bookwise-analytics/issues/15)                                | Heroku deployment failed due to missing packages    | Added sklearn to `pyproject.toml`           |


---

## 15. Quick Start

1. Clone repo:  
   `git clone https://github.com/larevolucia/bookwise-analytics.git`
2. Set up `.env` with API keys.
3. Install dependencies:  
   `pip install -e ".[dev,viz,ml]"`
4. Run notebooks for data prep and modeling.
5. Launch Streamlit:  
   `streamlit run app.py`
---

## Hugging Face Integration & Setup

This project uses **two Hugging Face repositories** for seamless data and model management:

- **Datasets & Plots:**  
  Repository type: `dataset`  
  Stores processed datasets and EDA plots for reproducibility and sharing.

- **Models:**  
  Repository type: `model`  
  Stores trained model artifacts for deployment and inference.

### 1. Create Hugging Face Accounts & Tokens

- Sign up at [Hugging Face](https://huggingface.co/join).
- Go to [Access Tokens](https://huggingface.co/settings/tokens) and create a **Write** token for each repository.
- Save both tokens in your `.env` file:
  ```
  HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  HUGGINGFACE_MODEL_TOKEN=hf_yyyyyyyyyyyyyyyyyyyyyyyyyyyy
  ```

### 2. Datasets & Plots Repository

- Clone or create a new dataset repo (e.g., `bookwise-analytics-ml`).
- Upload datasets and EDA plots using the [Hugging Face Hub CLI](https://huggingface.co/docs/hub/how-to-upload):
  ```bash
  huggingface-cli login --token $HUGGINGFACE_TOKEN
  huggingface-cli repo create bookwise-analytics-ml --type dataset
  huggingface-cli upload ./data/ --repo-type dataset --repo-id <your-username>/bookwise-analytics-ml
  ```

### 3. Models Repository

- Clone or create a new model repo (e.g., `popularity-score-model`).
- Upload model files:
  ```bash
  huggingface-cli login --token $HUGGINGFACE_MODEL_TOKEN
  huggingface-cli repo create popularity-score-model --type model
  huggingface-cli upload ./models/ --repo-type model --repo-id <your-username>/popularity-score-model
  ```

### 4. Using Hugging Face in Code

- Use the `datasets` library to load datasets and the `huggingface_hub` library to download model artifacts directly:

  ```python
  # Load dataset from Hugging Face Hub
  from datasets import load_dataset
  dataset = load_dataset("revolucia/bookwise-analytics-ml")

  # Download a specific model file from the model repo
  from huggingface_hub import hf_hub_download
  model_path = hf_hub_download(
      repo_id="revolucia/popularity-score-model",
      filename="modeling_data/et_model.pkl"
  )
  ```

- Adjust the `filename` argument to match the actual path in the model repo (e.g., `"modeling_data/et_model.pkl"`).

- For more details, see the [Hugging Face Hub documentation](https://huggingface.co/docs/hub/index).

---

### Google Books API

This project uses the Google Books API to enrich book metadata (pages, publisher, publication date, description, categories, etc.) for titles missing information after merging core datasets.

**Setup:**

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project ([guide](https://developers.google.com/workspace/guides/create-project)).
3. Enable the **Google Books API** (`APIs & Services > Library`).
4. Generate an **API key** (`APIs & Services > Credentials`).

Add the key and URLs to your `.env` file:

```bash
GOOGLE_BOOKS_API_KEY=<YOUR_KEY>
```

**Usage in notebooks:**
- The API is queried by ISBN (preferred) or by title/author for books without ISBNs.
- Results are cached in `data/raw/google_api_cache.json` to avoid duplicate requests and manage quota.
- See `03_Data_Enrichment_and_Dataset_Integration.ipynb` for implementation details.

---

### Open Library API

The Open Library API is used as the first enrichment source for missing metadata, since it is public and does not require authentication or API keys.

**Usage in notebooks:**
- Queried by ISBN for missing fields (pages, language, publisher, description, subjects).
- Results are cached in `data/raw/openlibrary_api_cache.json`.
- See `03_Data_Enrichment_and_Dataset_Integration.ipynb` for code and enrichment workflow.

---

## References


- [Regex101](https://regex101.com/): Online regex tester and debugger.
- [Text Cleaning in Python](https://pbpython.com/text-cleaning.html): A guide on cleaning text data using Python.
- [Pandas Documentation](https://pandas.pydata.org/docs/): datetime, combine_first,
- [NumPy](https://numpy.org/doc/stable/): exponential, logarithm, arange
- [DateUtils Documentation](https://dateutil.readthedocs.io/): for advanced date parsing.
- [TQDM Documentation](https://tqdm.github.io/): for progress bars in loops.
- [Requests Documentation](https://requests.readthedocs.io/en/latest/): for making HTTP requests.
- [Pandas Merging](https://pandas.pydata.org/docs/user_guide/merging.html): for combining DataFrames.
- [Scikit-Learn Documentation](https://scikit-learn.org/stable/documentation.html): for machine learning algorithms and evaluation metrics.
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): for consistent commit messages.
- On Gaussian Distribution: [Free Code Camp](https://www.freecodecamp.org/news/how-to-explain-data-using-gaussian-distribution-and-summary-statistics-with-python/), [Quantinsti](https://blog.quantinsti.com/gaussian-distribution/), [GeeksForGeeks Machine Learning](https://www.geeksforgeeks.org/machine-learning/gaussian-distribution-in-machine-learning/), [eeksForGeeks Python](https://www.geeksforgeeks.org/python/python-normal-distribution-in-statistics/), [PennState College](https://online.stat.psu.edu/stat857/node/77/)
- On Binnin Data: [GeeksForGeeks](https://www.geeksforgeeks.org/numpy/binning-data-in-python-with-scipy-numpy/)
- [Sentence Transformers](https://www.sbert.net/docs/quickstart.html): for generating text embeddings.
- [Pytest Documentation](https://docs.pytest.org/en/7.3.x/): for testing framework in Python.
- [geeksforgeeks.org: Combinations](https://www.geeksforgeeks.org/python/itertools-combinations-module-python-print-possible-combinations/): for generating combinations.
- [Scikit-learn](https://scikit-learn.org/): for ML models and pipelines.
- [Displayr: Learn What Are Residuals](https://www.displayr.com/learn-what-are-residuals/)
- [Medium: Understanding Residual Analysis in Regression](https://medium.com/@jangdaehan1/understanding-residual-analysis-in-regression-a-deep-dive-bc9ba6f3506d)
- [GeeksForGeeks: Residual Analysis](https://www.geeksforgeeks.org/maths/residual-analysis/)
- [Introduction to SHAP Values for Machine Learning Interpretability](https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability)
- [SHAP Documentation](https://shap.readthedocs.io/en/latest/)
- [Hugging face](https://huggingface.co/docs)
- [Open Library API documentation](https://openlibrary.org/developers/api)
- [Google Books API documentation](https://developers.google.com/books/docs/v1/using)
- ChatGPT: to refine and correct grammar textual explanations in README.md and notebooks.
- NotebookLM: Learning guide on data cleaning, to help me find next steps without providing any code.