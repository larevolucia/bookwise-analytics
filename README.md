# Bookwise Analytics: Book Subscription Optimization

CRISP-DM steps docummented at: [Project board](https://github.com/users/larevolucia/projects/15/views/1)
Live App: [Streamlit Dashboard](https://bookwise-analytics-1d891f772a24.herokuapp.com/)

## Business Understanding

### Context

This project simulates a subscription-based book club where members receive one monthly “credit” to select a book from a curated catalog. Despite stable subscriber numbers, engagement and redemption rates are declining, often due to poor book-member matches.

The goal is to transition from intuition-driven curation to data-driven selection, using predictive analytics to increase satisfaction, loyalty, and catalog diversity.

---

### Problem Statement

The business faces a growing inactive user base. While the library continues to expand, users struggle to find content that resonates with their tastes, leading to disengagement and eventual cancellation.

This project aims to move from intuition-based book selection to a **predictive recommendation system** that optimizes member satisfaction and long-term retention.

---

### Business Objectives

1. **Identify** the book and genre characteristics most strongly associated with member engagement.
2. **Predict** which titles are likely to achieve high satisfaction and retention potential.
3. **Simulate** potential retention uplift if algorithmic recommendations replace (or supplement) editorial curation.
4. **Safeguard** genre diversity and fairness in all recommendation outputs.

---

### Analytical Goals

* Use **Best Books Ever** and **Goodbooks-10k** datasets to emulate internal catalog data and user-behavioral data.
* Build a **regression-based predictive model** (with optional clustering) that estimates book engagement potential.
* Simulate the potential **retention uplift** from model-driven versus editorial selections.
* Deploy an **interactive Streamlit dashboard** that allows stakeholders to explore insights, prediction outputs, and diversity metrics.

---

### Business Requirements

| ID       | Business Requirement                                                                      | Success Indicator                                  | Dataset(s)             |
| -------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------- |
| **BR-1** | Identify which metadata and external features correlate with higher engagement.                  | Correlation ≥ 0.4 between features and engagement. | BBE            |
| **BR-2** | Predict which titles are most likely to achieve high engagement based on historical data. | Model RMSE < 1.0 or R² > 0.7.                      | BBE, Goodbooks |
| **BR-3** | Estimate potential retention uplift from algorithmic vs manual (editorial) selection.     | Simulated uplift ≥ 10%.                            | BBE, Goodbooks         |
| **BR-4** | Maintain diversity and fairness in recommendations across genres.                         | Shannon Entropy ≥ editorial baseline                 | BBE, Goodbooks                    |

---

### Hypotheses

| ID     | Hypothesis                                                                                                      | Validation Method                                                     | Expected Outcome                                                              |
| ------ | --------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **H1** | Books with high cross-platform ratings (>4.0) and multi-genre tags achieve higher engagement.                   | Correlation and multiple regression.                                  | Positive correlation (r > 0.4).                                               |
| **H2** | Historical rating and review patterns can predict engagement with ~80% accuracy.                                | Regression or hybrid recommender (XGBoost / collaborative filtering). | Model achieves RMSE < 1.0 or R² > 0.7.                                        |
| **H3** | Recent publications yield higher satisfaction. | Feature importance analysis and correlation with publication decade. | Publication recency (e.g., 2010s) appears among top features, but external engagement metrics are stronger predictors; recency has moderate impact. |
| **H4** | Algorithmic selection based on predicted engagement increases overall engagement by at least 10% compared to editorial or random selection. | Simulated uplift modeling using a logistic engagement–retention proxy. | ≥10% uplift in simulated engagement (proxy for retention). |

---

### Expected Business Impact

* **Recommendation:** Users are more likely to spend credits and remain active when recommendations align with predicted performance.
* **Retention:** Data-driven selection is projected to reduce churn by highlighting titles with higher predicted satisfaction.
* **Efficiency:** Editors can focus on curating top-performing genres or niche categories identified by analytics.
* **Scalability:** The recommendation model can be extended to new releases without manual review.

> **Note:** Actual subscriber-level retention data is unavailable. Therefore, predicted engagement scores are used as a proxy for retention likelihood, and the uplift metric estimates potential retention improvement through higher engagement.

---

## Datasets

This project integrates multiple **publicly available sources** to emulate both **internal system activity** and **catalog-wide data** of a subscription book service.

| Dataset                           | Source                                                                       | Purpose                                                                         |
| --------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Best Books Ever (BBE)**         | [Zenodo / GitHub](https://github.com/scostap/goodreads_bbe_dataset)          | Catalog-level metadata: ratings, votes, bbeScore     |
| **Goodbooks-10k**                 | [GitHub – zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) | User-behavior dataset: 6 M ratings from 53 K users
| **Overlap (BBE ∩ Goodbooks)**     | Derived (8 K books; ≈ 80 % of BBE, 15 % of Goodbooks)                        | Enables metadata linking  |
| **Open Library API** | [Open Library API](https://openlibrary.org/developers/api)                      | Fetches commercial or descriptive metadata.             |
| **Google Books API** | [Google Books API](https://developers.google.com/books)                      | Fetches commercial or descriptive metadata.             |

> **Why these sources:** These datasets together capture book quality (Goodreads ratings), user behavior (Goodbooks), and market context (Google API), forming a realistic simulation environment.

---

## Data Visualization & ML Tasks

| Business Requirement                             | Analytical / Visualization Task                                     | Expected Outcome                                    |
| ------------------------------------------------ | ------------------------------------------------------------------- | --------------------------------------------------- |
| **BR-1: Engagement Drivers**         | Correlation heatmaps; genre–rating scatter; feature-importance bars | Identify strongest predictors of satisfaction |
| **BR-2: High-Engagement Prediction**         | Train/test regression or hybrid recommender; report RMSE & R²       | Predict engagement score for unseen titles          |
| **BR-3: Uplift Estimation**| Simulate model vs. random and popularity baselines | ≥ 10% uplift in mean predicted engagement score |
| **BR-4: Diversity Fairness**          | Compute genre entropy and share of recommendations               | Balanced representation across genres               |

**Baselines for BR-3:**

* **Random baseline**: uniform K-sampling.
* **Popularity baseline**:  top-K by `numRatings`.
* **Model strategy**: top-K by **predicted engagement**.

---

## Minimum Viable Product (MVP)

| Goal                                  | Visual / Task                                                                                                                   | Outcome                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Assess data quality & distribution   | Histograms, missing-value matrices, pairplots                                                                                    | Identify cleaning needs and feature gaps                            |
| Analyse engagement vs rating    | Scatter/box plots by genre, correlation heatmap                                        | Quantify relationship strength                                    |
| Build baseline model                  | Popularity baseline + simple regression                                                              | Establish predictive benchmark
| Compare recommendation strategies | Display side-by-side lists for **Model**, **Editorial**, and **Random** selections | Demonstrate uplift engagement |
| Evaluate diversity                     | Genre entropy & catalog coverage                                                                                                | Ensure fair representation across genres       |


### Stretch Goals

| Enhancement                      | Description                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Engagement Uplift Simulation** | Extend the dashboard with interactive Streamlit controls (top-K slider, baseline selector, weighting options) to visualize engagement uplift simulations.  |
| **NLP/Google API Integration**   | Enrich the dataset with additional textual and descriptive metadata for hybrid content-based modelling (e.g., genre embeddings, summaries)                                                       |
| **Collaborative filtering**       | Implement a user–book recommendation model (e.g., `surprise` or `lightfm`) using Goodbooks rating data to predict unseen preferences and compare its results with regression-based predictions.                                  |
| **Caching & Optimization**       | Add model caching and session-based optimization to improve dashboard responsiveness, especially for large datasets or repeated predictions.                                  |

---

## ML Business Case

**Business Objective:**
Estimate the potential uplift in engagement and retention achievable through a predictive recommendation system versus manual selection.

**Primary Modelling Scope (MVP):**

* **Regression models:** (RandomForest, XGBoost, CatBoost) trained on book-level features such as ratings, reviews, and metadata from the BBE dataset to predict expected engagement.
* **User Cluster:** Apply unsupervised learning to group readers by rating behavior or genre preference, enabling semi-personalized book curation for each cluster.

**Stretch / Future Extensions**

* **Collaborative filtering:** Implement memory- or model-based recommender systems (using `surprise` or `lightfm`) to learn latent user–item patterns from Goodbooks data and deliver personalized recommendations.
* **NLP analysis:** Use text analysis (`nltk`, `spacy`) to process book descriptions or summaries, generating embedding-level features for hybrid recommendation.

**Python / ML Libraries (tentative):**
`pandas`, `numpy`, `scikit-learn`, `surprise`, `lightfm`, `xgboost`, `nltk`/`spacy` *(optional)*, `plotly`, `streamlit`

**Evaluation Metrics (offline simulation):**

### **Evaluation Metrics (Offline Simulation)**

| Metric                        | Purpose                                                                                  | Linked Business Requirement |
| ----------------------------- | ---------------------------------------------------------------------------------------- | --------------------------- |
| **RMSE / R²**                 | Model accuracy for engagement prediction                                                 | BR-2                        |
| **Simulated Uplift (%)**      | Average engagement gain vs. random and popularity baselines (proxy for retention uplift) | BR-3                        |
| **Diversity Index (Entropy)** | Genre fairness and diversity within recommendations                                      | BR-4                        |
| **Precision@K / HitRate@K**   | *(Optional)* Measures relevance of top-N predicted recommendations                       | BR-2 / Stretch              |

---

### Notebook Structures

```
/notebooks
├── 01_Data_Collection.ipynb        # initial exploration and completeness check of all datasets
├── 02_Data_Cleaning.ipynb          # regex cleaning, NaN handling, and type conversions
├── 03_Data_Imputation.ipynb        # external API enrichment and dictionary-based caching
├── 04_Exploratory_Data_Analysis.ipynb  # EDA: correlations, trends, and feature relationships
├── 05_Feature_Engineering.ipynb    # feature creation (engagement_score, recency, popularity)
├── 06_Modeling.ipynb               # regression, clustering, and recommendation model development
```

> Each notebook represents a phase of the **CRISP-DM** process:
>
> * **Data Collection (LO7)** – explores and audits dataset quality and structure.
> * **Data Cleaning (LO7.2)** – standardizes and validates fields for analytical use.
> * **Data Imputation (LO7.1)** – retrieves missing values via APIs while ensuring reproducibility.
> * **Exploratory Data Analysis (LO4)** – investigates feature relationships, correlations, and trends.
> * **Feature Engineering (LO4 & LO5)** – generates predictive features tied to business hypotheses.
> * **Modeling (LO5)** – builds and evaluates machine learning pipelines to meet business KPIs.
>
> This modular organization maintains reproducibility and transparency from raw data to model evaluation, meeting the **Code Institute’s CRISP-DM and Learning Outcome** standards for the Predictive Analytics project.

---
## Model Dataset Integration Flowchart

                ┌────────────────────┐
                │   GOODBOOKS DATA   │
                │ (cleaned features) │
                └─────────┬──────────┘
                          │
              merge on `goodreads_id_clean`
                          │
                ┌─────────▼──────────┐
                │   BBE DATASET       │
                │ (rename cols to     │
                │  prevent conflicts) │
                └─────────┬──────────┘
                          │
                          ▼
               ┌────────────────────────┐
               │   MERGED MASTER DATA   │
               └─────────┬──────────────┘
                         │
       ┌─────────────────┴───────────────────┐
       │                                     │
       ▼                                     ▼
┌──────────────────────┐           ┌────────────────────────┐
│  DATASET A:          │           │  DATASET B:             │
│  Cross-Platform      │           │  Clean Modeling         │
│  Validation          │           │  (No BBE behavioral     │
│  • keeps BBE         │           │   features)             │
│    behavioral data   │           │  • avoids leakage       │
│  • used only for     │           │  • supports cold start  │
│    insight studies   │           │    scenarios            │
└──────────────────────┘           └────────────────────────┘

---

## Dashboard Design (Streamlit MVP)

| Page                                                              | Purpose                                                                                              | Key Visuals & Elements                                                                                                     |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **1. Executive Summary**                                          | Show KPIs (RMSE, R², engagement score averages).                                                     | Metric cards + bar chart of model vs baseline.     |
| **2. Book Analytics Explorer**                                    | Explore correlations, trends, and diversity metrics.                                                 |  Plotly scatter + heatmap filters + genre diversity visualizations. |
| **3. Recommendation Comparison**                                  | Compare Model / Editorial / Random strategies.                                                       | Side-by-side tables + bar chart of predicted scores.  |
| **4. Model Runner: Top 10 Books**                                 | Run the trained model on supply data and show top 10 highly rated books.                             | Table of top 10 books by predicted engagement score. |

---

## MVP vs. Stretch Scope

| Category       | MVP                                                                                                                               | Stretch                                                                                                   |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Data**       | Use **BBE** and **Goodbooks-10k** datasets for engagement and behavioral signals (8 K overlap subset).                            | Integrate **Google Books API** for additional commercial and descriptive metadata.                        |
| **ML**         | Implement a **regression model** to generate predicted engagement scores.                     | **Hybrid model** with text embeddings or CF features.  |
| **Visuals**    | Four Streamlit pages listed above. | Add interactive **Uplift Simulator**. |
| **Evaluation** | RMSE / R² + genre diversity entropy.     | Feature-importance & Precision@K plots.   |
| **Deployment** | Streamlit Cloud or Heroku with static data + saved model.  | Add caching and optimization.      |
---
## Key Metrics Explained

**Simulated Uplift Calculation:**  
Simulated uplift measures the percentage increase in the average predicted engagement score when switching from editorially selected books to those recommended by the predictive model. It is calculated by comparing the mean predicted scores for both groups, indicating whether model-driven recommendations are expected to outperform traditional selections.

**Genre Entropy Calculation:**  
Genre entropy quantifies the diversity of genres among recommended books using Shannon entropy. A higher entropy value reflects a more varied and balanced genre distribution, helping ensure fairness and representation across recommendations.

**Popularity Score:**  
Popularity score is a composite metric created by standardizing and summing key engagement signals such as log-transformed review counts, log-transformed rating counts, and average rating. This score captures both the visibility and engagement of a book, making it a robust target for predictive modeling.

**External Popularity Score:**  
External popularity score combines standardized external signals, such as ratings, number of ratings, votes, engagement scores, and liked percentage, from sources outside the internal catalog. By aggregating these features, it provides a holistic measure of a book's popularity and appeal in the broader market.

---
# Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/larevolucia/bookwise-analytics.git
   cd bookwise-analytics
   ```

2. **Create a `.env` file** in the project root with your Google API Key and Hugging Face tokens:
   ```
   GOOGLE_BOOKS_API_KEY=your_google_books_api_key
   HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   HUGGINGFACE_MODEL_TOKEN=hf_yyyyyyyyyyyyyyyyyyyyyyyyyyyy
   ```

3. **Install dependencies** (see Local Development section).

4. **Datasets and models** 
The original datasets are large and not included in the repo, however, they are open-source and can be downloaded from their respective sources (links in Datasets section).

You can run the Jupyter notebooks to fetch original datasets and processed datasets. 

[Datasets and Plots](https://huggingface.co/datasets/revolucia/bookwise-analytics-ml/tree/main)
[Deployed Models](https://huggingface.co/revolucia/popularity-score-model/tree/main)

5. **Run the Streamlit app locally:**
   ```bash
   streamlit run app.py
   ```

6. **Deploy to Heroku:**  
   Follow the steps in the Heroku Deployment section.  
   Make sure to set both Hugging Face tokens as Heroku config vars.
  
## Local Development

To work locally (notebooks, utilities, ML), install via the project’s `pyproject.toml`. Avoid requirements.txt; it is only for Heroku.

1) Create and activate a virtual environment:
- macOS/Linux:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  ```
- Windows:
  ```bash
  py -m venv .venv && .venv\Scripts\activate
  ```

2) Install the package with extras you need:
- Core + notebooks + dev tools:
  ```bash
  pip install -U pip
  pip install -e ".[dev,viz]"
  ```
- Add analytics:
  ```bash
  pip install -e ".[analytics]"
  ```
- Add ML/NLP (CPU-only, large):
  ```bash
  pip install -e ".[ml]"
  ```
- Add text matching utilities:
  ```bash
  pip install -e ".[text]"
  ```

3) Run tests:
```bash
pytest
```

4) Launch notebooks:
```bash
python -m ipykernel install --user --name bookwise-analytics
jupyter notebook
```

> Do not use requirements.txt for local installs. It is intentionally minimal for Heroku deployment.

## Hugging Face Integration

This project uses **two Hugging Face repositories**:

- `bookwise-analytics-ml`: stores datasets and EDA plots.
- `popularity-score-model`: stores trained model artifacts.

**Important:**  
- You must create a separate access token for each repository.
- Save both tokens in your `.env` file:
  ```
  HUGGINGFACE_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx
  HUGGINGFACE_MODEL_TOKEN=hf_yyyyyyyyyyyyyyyyyyyyyyyyyyyy
  ```
- Use the correct token when uploading or downloading files for each repo.

## Heroku Deployment

The Streamlit app on Heroku uses the minimal runtime libs defined in requirements.txt and the Procfile.

Prerequisites:
- Heroku CLI installed and logged in.
- A Heroku app created.

Deploy steps:
```bash
heroku git:remote -a <your-heroku-app-name>
git add -A
git commit -m "Deploy Streamlit app"
git push heroku main
```

Heroku will:
- Install dependencies from requirements.txt.
- Use Procfile:
  `web: sh setup.sh && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
- Generate Streamlit config via setup.sh.

Environment variables:
- Place any `.env` keys into Heroku config vars:
```bash
heroku config:set YOUR_ENV_KEY=your_value
```

Troubleshooting:
- Check logs:
```bash
heroku logs --tail
```


---
# References

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
- ChatGPT: to refine and correct grammar textual explanations in README.md and notebooks.
- NotebookLM: Learning guide on data cleaning, to help me find next steps without providing any code.

