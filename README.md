# Bookwise Analytics: Predictive Book Recommendation System

Live App: [Streamlit Dashboard](https://bookwise-analytics-1d891f772a24.herokuapp.com/)

Project Repo: [GitHub Repository](https://github.com/users/larevolucia/projects/15)

LinkedIn: [Project Inception Post](https://www.linkedin.com/pulse/learning-think-like-data-scientist-l%C3%BAcia-reis-yca6e/?trackingId=PuHDRjSOpaBrIQ5TJOR%2Bsw%3D%3D)

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Understanding](#2-business-understanding)
  - [Problem Statement](#21-problem-statement)
  - [Business Objectives](#22-business-objectives)
  - [Stakeholder Benefits](#23-stakeholder-benefits)
3. [Business Requirements & Mapping](#3-business-requirements--mapping)
  - [3.1 User Stories](#31-user-stories)
  - [3.2 Mapping to ML & Visualization](#32-mapping-to-ml--visualization)
  - [3.3 Requirements Table](#33-requirements-table)
  - [3.4 Stretch Goal: Clustering](#34-stretch-goal-clustering)
4. [Hypotheses & Validation](#4-hypotheses--validation)
5. [Datasets](#5-datasets)
6. [Data & Model Artefacts](#6-data--model-artefacts)
7. [Data Collection & Preparation](#7-data-collection--preparation)
8. [Analytical & ML Tasks](#8-analytical--ml-tasks)
9. [ML Business Case](#9-ml-business-case)
10. [Dashboard Design](#10-dashboard-design)
11. [Project Structure & Notebooks](#11-project-structure--notebooks)
12. [Deployment & Local Development](#12-deployment--local-development)
13. [Model Evaluation & Business Impact](#13-model-evaluation--business-impact)
14. [References & Attribution](#14-references--attribution)
15. [Bug Fixes](#15-bug-fixes)
16. [Test & Coverage](#16-test--coverage)
17. [Quick Start](#17-quick-start)
18. [Hugging Face Integration & Setup](#18-hugging-face-integration--setup)
  - [Create Accounts & Tokens](#181-create-accounts--tokens)
  - [Datasets & Plots Repo](#182-datasets--plots-repo)
  - [Models Repo](#183-models-repo)
  - [Using HF in Code](#184-using-hf-in-code)
19. [Google Books API](#19-google-books-api)
20. [Open Library API](#20-open-library-api)
21. [References](#21-references)

---

## 1. Project Overview

**Purpose:**  
Bookwise Analytics is a data-driven recommendation system for a subscription-based book club. The goal is to optimize book selection and user engagement using Machine Learning (ML), replacing intuition-based curation with predictive analytics. The project delivers a Streamlit dashboard for stakeholders to explore insights, model outputs, and diversity metrics.

**Target Audience:**  
Business stakeholders, data practitioners, and editorial teams seeking to maximize user satisfaction and retention in a book subscription service.

---

## 2. Business Understanding

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

## 3. Business Requirements & Mapping

| ID   | Requirement                                                                 | Success Indicator                        | Dataset(s)         | Linked Dashboard Page      |
|------|-----------------------------------------------------------------------------|------------------------------------------|--------------------|---------------------------|
| BR-1 | Identify features correlated with engagement                                | Correlation ≥ 0.4                        | BBE                | Analytics Explorer        |
| BR-2 | Predict high-engagement titles                                              | Model RMSE < 1.0 or R² > 0.7             | BBE, Goodbooks     | Model Runner              |
| BR-3 | Estimate retention uplift from recommendations                              | Simulated uplift ≥ 10%                   | BBE, Goodbooks     | Recommendation Comparison |
| BR-4 | Maintain diversity/fairness in recommendations                             | Shannon Entropy ≥ editorial baseline     | BBE, Goodbooks     | Diversity Metrics         |

---

## 3.1. User Stories

The following user stories are implemented and tracked via GitHub issues.  
Each story includes ML tasks, actions, and acceptance criteria.

**[High Engagement Titles](https://github.com/larevolucia/bookwise-analytics/issues/21)**  
As an editorial team member, I want to see which books are predicted to have high engagement, so I can focus curation efforts.

**[Engagement Uplift Prediction](https://github.com/larevolucia/bookwise-analytics/issues/18)**  
As a business stakeholder, I want to compare editorial vs. model-driven recommendations to understand uplift, so I can make informed decisions.

**[Feature Importance for Engagement](https://github.com/larevolucia/bookwise-analytics/issues/16)**  
As a business stakeholder, I want to understand which book features drive engagement, so I can optimize catalog selection.

**[Genre Fairness](https://github.com/larevolucia/bookwise-analytics/issues/19)**  
As a stakeholder, I want to ensure recommendations maintain genre diversity and fairness, so I don't alienate any user segments.

**[Summary Dashboard](https://github.com/larevolucia/bookwise-analytics/issues/20)**  
As a stakeholder, I want an executive summary page showing KPIs and project overview, so I can quickly assess performance.

**[Title Acquisition](https://github.com/larevolucia/bookwise-analytics/issues/22)**  
As a user, I want to search for any book and see its predicted engagement score, so I can guide title acquisition decisions.

---

## 3.2. Mapping to ML & Visualization

| User Story                        | ML Task / Visualization                        | Actions Required                                                                                   |
|------------------------------------|-----------------------------------------------|----------------------------------------------------------------------------------------------------|
|  [High Engagement Titles](https://github.com/larevolucia/bookwise-analytics/issues/21)          | Engagement prediction, leaderboard            | Model scoring, leaderboard table                                              |
| [Engagement Uplift Prediction](https://github.com/larevolucia/bookwise-analytics/issues/18)       | Editorial vs. model uplift metric           | Display sets, calculate predicition for each set, calculate uplift                          |
| [Feature Importance for Engagement](https://github.com/larevolucia/bookwise-analytics/issues/16)  | Feature importance analysis                   | Train model, extract importances, visualize, actionable insights                         |
| [Genre Fairness ](https://github.com/larevolucia/bookwise-analytics/issues/19)                    | Genre diversity/fairness metrics              | Compute shares, entropy, visualize                                            |
| [Summary Dashboard](https://github.com/larevolucia/bookwise-analytics/issues/20)                  | Executive KPIs dashboard                      | Aggregate KPIs, overview, navigation                                             |
| [Title Acquisition](https://github.com/larevolucia/bookwise-analytics/issues/22)                  | Search + engagement prediction                | Search bar, engagement prediction          |

---

## 3.3. Requirements Table
| ID   | Requirement                                                                 | Success Indicator                        | Dataset(s)         | Linked Dashboard Page      |
|------|-----------------------------------------------------------------------------|------------------------------------------|--------------------|---------------------------|
| BR-1 | Identify features correlated with engagement                                | Correlation ≥ 0.4                        | BBE                | Analytics Explorer        |
| BR-2 | Predict high-engagement titles                                              | Model RMSE < 1.0 or R² > 0.7             | BBE, Goodbooks     | Model Runner              |
| BR-3 | Estimate retention uplift from recommendations                              | Simulated uplift ≥ 10%                   | BBE, Goodbooks     | Recommendation Comparison |
| BR-4 | Maintain diversity/fairness in recommendations                             | Shannon Entropy ≥ editorial baseline     | BBE, Goodbooks     | Diversity Metrics         |

> If these thresholds are not met, the corresponding ML task is considered unsuccessful and is not recommended for operational use.
---

## 3.4. Stretch Goal: Clustering

As an additional feature, this project implements **user clustering** using KMeans to segment members based on their reading behavior and preferences. This segmentation helps identify distinct user groups and supports more targeted marketing and personalization strategies.

#### Clustering Approach

- **Features Used:**  
  Aggregated user-level features such as average pages per book, number of genres read, genre diversity, genre concentration, top genre share, and number of interactions.
- **Preprocessing:**  
  Missing values are imputed (numerical: median, categorical: mode), categorical features are one-hot encoded, and all features are standardized.
- **Algorithm:**  
  KMeans clustering is applied to the processed features. The optimal number of clusters is determined using the silhouette score and elbow method.
- **Cluster Profiles:**  
  Analysis revealed **two main user segments**:
  - **Cluster 0: Genre Specialists**  
    - Fewer ratings overall  
    - Higher average rating per book  
    - Preference for newer and longer books  
    - Less genre diversity, more focused on a single genre  
  - **Cluster 1: Genre Explorers**  
    - More ratings overall  
    - Slightly lower average rating per book  
    - Preference for older and shorter books  
    - Higher genre diversity, less focused on a single genre  

#### Business Interpretation

- **Genre Specialists** may respond well to targeted recommendations within their favorite genres and new releases.
- **Genre Explorers** may appreciate diverse recommendations and discovery-oriented features.

#### Outputs

- Cluster assignments and profiles are available in the dashboard's "Member Insights" page.
- The clustering workflow, evaluation, and business rationale are fully documented in [`notebooks/07_Member_Cluster.ipynb`](notebooks/07_Member_Cluster.ipynb).

*This segmentation enables more personalized engagement strategies and actionable insights for marketing and editorial teams.*

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

## 6. Data & Model Artefacts

To ensure reproducibility and keep the repository lightweight, large datasets and trained model artefacts are hosted on Hugging Face.

#### Datasets
- **Bookwise Analytics – Modeling Dataset**  
  https://huggingface.co/datasets/revolucia/bookwise-analytics-ml  
  Cleaned and feature-engineered dataset used for engagement modeling.

- **Book Club User Clusters**  
  https://huggingface.co/revolucia/bookclub-cluster  
  Precomputed user segmentation results for member insights.

#### Models
- **Popularity Score Model (ExtraTreesRegressor)**  
  https://huggingface.co/revolucia/popularity-score-model  
  Trained regression model and evaluation metrics used by the Streamlit application.

---

## 7. Data Collection & Preparation

- **Data Collection:** Jupyter notebooks fetch and audit datasets, including API enrichment.
- **Data Cleaning:** Handle missing values, standardize fields, and merge datasets.
- **Feature Engineering:** Create popularity, metadata scores and log-transform skewed features.
- **Data Preparation:** Documented in `/notebooks/` (see structure below).

---

## 8. Analytical & ML Tasks

| Requirement | Task                                      | Notebook(s)                  | Outcome/Metric                |
|-------------|-------------------------------------------|------------------------------|-------------------------------|
| BR-1        | Correlation, feature importance           | 04_Exploratory_Data_Analysis | Key predictors identified     |
| BR-2        | Regression, hybrid recommender            | 06_Modeling                  | RMSE, R², MAE         |
| BR-3        | Uplift simulation                         | 06_Modeling, dashboard       | % uplift in engagement        |
| BR-4        | Genre entropy, diversity metrics          | 04_Exploratory_Data_Analysis | Entropy, genre coverage       |

---

## 9. ML Business Case

- **Aim:** Predict book engagement to optimize recommendations and retention.
- **Learning Method:** Regression (RandomForest, XGBoost) and clustering.
- **Success Metrics:** RMSE < 1.0, R² > 0.7, ≥10% uplift, high genre entropy.
- **Model Output:** Predicted engagement scores, top-K recommendations.
- **Relevance:** Directly supports business KPIs for engagement and retention.

---

## 10. Dashboard Design

**Pages:**

- **Executive Summary:** KPIs (RMSE, R², uplift), summary plots. *(BR-2, BR-3)*
- **Analytics Explorer:** Correlations, feature importance, genre diversity. *(BR-1, BR-4)*
- **Recommendation Comparison:** Model vs. editorial vs. random selections, uplift plots. *(BR-3)*
- **Model Runner:** Top 10 books by predicted engagement. *(BR-2)*
- **Diversity Metrics:** Genre entropy, coverage plots. *(BR-4)*

*Each page includes textual interpretation of plots and clear statements on model performance.*

---

## 11. Project Structure & Notebooks

```
/notebooks
├── 01_Data_Collection.ipynb
├── 02_Data_Cleaning.ipynb
├── 03_Data_Enrichment_and_Dataset_Integration.ipynb
├── 04_Exploratory_Data_Analysis.ipynb
├── 05_Feature_Engineering.ipynb
├── 06_Modeling.ipynb
├── 07_Member_Cluster.ipynb
```
- Each notebook starts with objectives, inputs, and outputs.
- Data preparation and feature engineering are clearly documented.

---

## 12. Deployment & Local Development

- **Streamlit app:** `streamlit run app.py`
- **Heroku deployment:** See instructions in this README.
- **Environment:** Python, pip, virtualenv, requirements.txt, Procfile, setup.sh maintained.
- **Version Control:** All code managed in GitHub with clear commit history.

---

## 13. Model Evaluation & Business Impact

### Summary

The Bookwise Analytics recommendation engine was evaluated using multiple regression models, with the **ExtraTreesRegressor** selected for deployment due to its strong performance:

- **Test R²:** 0.81
- **Test RMSE:** 0.95
- **Test MAE:** 0.57

These results exceed the business success criteria (RMSE < 1.0 or R² > 0.7), confirming the model's reliability for predicting book engagement.

**Feature importance analysis** shows that external popularity signals (e.g., number of ratings, votes, composite popularity score) are the strongest predictors, with publication recency and select metadata features providing additional value.

### Recommendation Uplift & Diversity

On the dashboard's Recommendation Comparison page, the model-driven approach achieved:

- **Simulated Uplift:** **42.13%** (vs. editorial selection, far exceeding the 10% target)
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

## 14. References & Attribution

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
| [File not found](https://github.com/larevolucia/bookwise-analytics/issues/24)                                | Path pointing to local file    | Hosted version uses hosted file.            |
| [Feature importance chart: Infinite extent warning](https://github.com/larevolucia/bookwise-analytics/issues/25) | Altair/Streamlit chart warning due to missing or invalid data in feature importance CSV | Data validation and user-friendly error handling added. Warning is harmless since charts display correctly, so further action was deferred. |
| [Vega-Lite compatibility](https://github.com/larevolucia/bookwise-analytics/issues/26) | Console warning due to Vega-Lite version mismatch between Altair (v5.x) and Streamlit frontend (v6.x). | All charts render and function as expected, so package upgrades were deferred to avoid unnecessary dependency complexity. The warning can be safely ignored unless future issues arise. |

## 16. Test & Coverage

- **Test results and coverage details are available in [`documentation/TEST.md`](documentation/TEST.md).**
- **Summary:**  
  All 88 tests pass successfully (`pytest`).  
  Overall code coverage is **41%**, with 100% coverage for core modeling pipeline code.  
  Most cleaning and feature engineering utilities are well covered.  
  Some analysis and EDA modules have low or no coverage—see the [full report](documentation/TEST.md) for details.

---

## 17. Quick Start

1. Clone repo:  
   `git clone https://github.com/larevolucia/bookwise-analytics.git`
2. Set up `.env` with API keys.
3. Install dependencies:  
   `pip install -e ".[dev,viz,ml]"`
> requirements.txt is optimized for deployment; use `pyproject.toml` for local development.
4. Run notebooks for data prep and modeling.
5. Launch Streamlit:  
   `streamlit run app.py`
---

## 18. Hugging Face Integration & Setup

This project uses **two Hugging Face repositories** for seamless data and model management:

- **Datasets & Plots:**  
  Repository type: `dataset`  
  Stores processed datasets and EDA plots for reproducibility and sharing.

- **Models:**  
  Repository type: `model`  
  Stores trained model artifacts for deployment and inference.

### 18.1. Create Accounts & Tokens

- Sign up at [Hugging Face](https://huggingface.co/join).
- Go to [Access Tokens](https://huggingface.co/settings/tokens) and create a **Write** token for each repository.
- Save both tokens in your `.env` file:
  ```
  HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  HUGGINGFACE_MODEL_TOKEN=hf_yyyyyyyyyyyyyyyyyyyyyyyyyyyy
  HUGGINGFACE_CLUSTER_TOKEN=hf_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzz
  ```

### 18.2. Datasets & Plots Repo

- Clone or create a new dataset repo (e.g., `bookwise-analytics-ml`).
- Upload datasets and EDA plots using the [Hugging Face Hub CLI](https://huggingface.co/docs/hub/how-to-upload):
  ```bash
  huggingface-cli login --token $HUGGINGFACE_TOKEN
  huggingface-cli repo create bookwise-analytics-ml --type dataset
  huggingface-cli upload ./data/ --repo-type dataset --repo-id <your-username>/bookwise-analytics-ml
  ```

### 18.3. Models Repo

- Clone or create a new model repo (e.g., `popularity-score-model`).
- Upload model files:
  ```bash
  huggingface-cli login --token $HUGGINGFACE_MODEL_TOKEN
  huggingface-cli repo create popularity-score-model --type model
  huggingface-cli upload ./models/ --repo-type model --repo-id <your-username>/popularity-score-model
  ```

### 18.4. Using HF in Code

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

## 19. Google Books API

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

## 20. Open Library API

The Open Library API is used as the first enrichment source for missing metadata, since it is public and does not require authentication or API keys.

**Usage in notebooks:**
- Queried by ISBN for missing fields (pages, language, publisher, description, subjects).
- Results are cached in `data/raw/openlibrary_api_cache.json`.
- See `03_Data_Enrichment_and_Dataset_Integration.ipynb` for code and enrichment workflow.

---

## 21. References


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
- [Testing Models with Pytest](https://www.fuzzylabs.ai/blog-post/the-art-of-testing-machine-learning-pipelines)
- ChatGPT: to refine and correct grammar textual explanations in README.md and notebooks.
- NotebookLM: Learning guide on data cleaning, to help me find next steps without providing any code.

## Acknowledgements

Thanks to all who provided feedback and support during this project, including peers, mentors, and the open-source community.

## Potential Next Steps

**Cold Start Model for Metadata Features Importance:**

Develop a model to estimate engagement or recommendation quality using only metadata features (e.g., genre, author, publication year) for new books with no external popularity signals (ratings, reviews, etc.).

**Hybrid Model for Collaborative Filtering:**

Explore a hybrid approach that combines collaborative filtering (user-item interactions) with content-based features to improve recommendations, especially for users or books with sparse data.

**New Feature Engineering:**

Investigate additional features such as text embeddings from book descriptions, author popularity or publisher reputation via wikipedia or social media signals to enhance model performance.

**Best Seller Data Integration:**

Incorporate external best seller lists (e.g. NYT, Amazon) as features to capture market trends and further improve engagement predictions.