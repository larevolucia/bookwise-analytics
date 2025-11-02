
# Bookwise Analytics: Book Subscription Optimization

CRISP-DM steps docummented at: [Project board](https://github.com/users/larevolucia/projects/15/views/1)

## Business Understanding

### Context

This project simulates a subscription-based book club where members receive one monthly “credit” to select a book from a large catalog. Despite stable subscriber numbers, engagement and redemption rates are declining, often due to poor book-member matches.

This project explores how a subscription-based book club can optimize member engagement and retention through data-driven book recommendations.

The aim is to transition from manual, intuition-based book curation to an **evidence-based selection system** that maximizes satisfaction, loyalty, and catalog diversity.

---

### Problem Statement

The business faces a growing inactive user base. While the library continues to expand, users struggle to find content that resonates with their tastes, leading to disengagement and eventual cancellation.

This project aims to move from intuition-based book selection to a **predictive recommendation system** that optimizes member satisfaction and long-term retention.

---

### Business Objectives

1. **Understand** what book features (e.g. genre, ratings, price, reviews, publication recency) are most correlated with member engagement.
2. **Predict** which titles are likely to achieve high satisfaction and retention potential.
3. **Simulate** potential retention uplift if algorithmic recommendations replace (or supplement) editorial curation.
4. **Ensure** the recommendation system maintains diversity and fairness across genres.

---

### Analytical Goals

* Use **Best Books Ever** and **Goodbooks-10k** datasets to simulate internal catalog data and member behavior, respectively.
* Build a **hybrid machine learning model** that predicts book satisfaction using historical patterns in ratings and reviews.
* Create a **synthetic retention simulation** to estimate the potential uplift in active user retention when the model is applied.
* Deploy an **interactive Streamlit dashboard** that allows stakeholders to explore insights, prediction outputs, and diversity metrics.

---

### Business Requirements

| ID       | Business Requirement                                                                      | Success Indicator                                  | Dataset(s)             |
| -------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------- |
| **BR-1** | Identify which book and genre features correlate with higher engagement.                  | Correlation ≥ 0.4 between features and engagement. | BBE            |
| **BR-2** | Predict which titles are most likely to achieve high engagement based on historical data. | Model RMSE < 1.0 or R² > 0.7.                      | BBE, Goodbooks |
| **BR-3** | Estimate potential retention uplift from algorithmic vs manual (editorial) selection.     | Simulated uplift ≥ 10%.                            | BBE, Goodbooks         |
| **BR-4** | Maintain diversity and fairness in recommendations across genres.                         | Shannon Entropy ≥ baseline (0.7).                  | All                    |

---

### Hypotheses

| ID     | Hypothesis                                                                                                      | Validation Method                                                     | Expected Outcome                                                              |
| ------ | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **H1** | Books with high cross-platform ratings (>4.0) and multi-genre tags achieve higher engagement.                   | Correlation and multiple regression.                                  | Positive correlation (r > 0.4).                                               |
| **H2** | Historical rating and review patterns can predict engagement with ~80% accuracy.                                | Regression or hybrid recommender (XGBoost / collaborative filtering). | Model achieves RMSE < 1.0 or R² > 0.7.                                        |
| **H3** | Recent publications yield higher satisfaction.                               | Correlation and time-series analysis.                                 | Negative correlation between book publication date and satisfaction. |
| **H4** | Algorithmic selection based on predicted engagement increases overall engagement by at least 10% compared to editorial or random selection. | Simulated uplift modeling using a logistic engagement–retention proxy. | ≥10% uplift in simulated engagement (proxy for retention). |

---

### Expected Business Impact

* **Personalization:** Users are more likely to spend credits and remain active when recommendations align with preferences.
* **Retention:** Data-driven selection is projected to reduce churn by highlighting titles with higher predicted satisfaction.
* **Efficiency:** Editors can focus on curating top-performing genres or niche categories identified by analytics.
* **Scalability:** The recommendation model can be extended to new releases without manual review.

> **Note:** Actual subscriber-level retention data is unavailable. Therefore, predicted engagement scores are used as a proxy for retention likelihood, and the uplift metric estimates potential retention improvement through higher engagement.

---

## Datasets

This project integrates multiple **publicly available sources** to emulate both **internal system activity** and **catalog-wide data** of a subscription book service.

| Dataset                           | Source                                                                       | Purpose                                                                         |
| --------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Best Books Ever (BBE)**         | [Zenodo / GitHub](https://github.com/scostap/goodreads_bbe_dataset)          | Simulates internal catalog performance; includes bbeScore, votes, ratings.      |
| **Goodbooks-10k**                 | [GitHub – zygmuntz/goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k) | Simulates user behavior: 6 M ratings from 53 K users (collaborative filtering). |
| **Overlap (BBE ∩ Goodbooks)**     | Derived (8 K books; ≈ 80 % of BBE, 15 % of Goodbooks)                        | Enables cross-dataset linking and unified metadata.                             |
| *(tentative)* **Google Books API** | [Google Books API](https://developers.google.com/books)                      | Fetches commercial or descriptive metadata.             |

> **Why these sources:** These datasets together capture book quality (Goodreads ratings), user behavior (Goodbooks), and market context (Google API), forming a realistic simulation environment.

Each dataset will be cleaned and unified into a consistent schema *(tentative)* (`book_id`, `title`, `author`, `genre`, `rating`, `reviews`, `popularity_score`, `price`, `publish_year`, `user_id`)

---

## Data Visualization & ML Tasks

| Business Requirement                             | Analytical / Visualization Task                                     | Expected Outcome                                    |
| ------------------------------------------------ | ------------------------------------------------------------------- | --------------------------------------------------- |
| **BR-1: Identify drivers of engagement**         | Correlation heatmaps; genre–rating scatter; feature-importance bars | Quantify which features most influence satisfaction |
| **BR-2: Predict high-engagement titles**         | Train/test regression or hybrid recommender; report RMSE & R²       | Predict engagement score for unseen titles          |
| **BR-3: Estimate engagement uplift** *(simulation)* | Algorithmic uplift simulation comparing model vs. random and popularity baselines | ≥ 10% uplift in mean predicted engagement score (proxy for retention) |
| **BR-4: Maintain diversity & fairness**          | Genre entropy, share-of-recommendations by genre                    | Balanced representation across genres               |

**Baselines for BR-3:**

* **Random baseline**: sample K titles uniformly.
* **Popularity baseline**: pick top-K by `numRatings` (BBE) or `Reviews` (Amazon).
* **Model strategy**: pick top-K by **predicted engagement**.

---

## Minimum Viable Product (MVP)

| Goal                                  | Visual / Task                                                                                                                   | Outcome                                                           |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Explore data quality & distribution   | Histograms, missing-data matrices, pairplots                                                                                    | Identify cleaning needs & feature gaps                            |
| Show engagement vs rating patterns    | Scatter/box by genre; correlation heatmap                                                                                       | Quantify relationship strength                                    |
| Build baseline model                  | Popularity baseline + simple collaborative filtering or regression                                                              | Benchmark for predictive model                                    |
| Compare recommendation strategies | Display side-by-side lists for **Model**, **Editorial**, and **Random** selections; show mean predicted engagement per strategy | Demonstrate model use in-app and evaluate engagement uplift proxy |
| Measure diversity                     | Genre entropy & catalog coverage                                                                                                | Detect narrow vs broad recommendations and ensure fairness        |


### Stretch Goals

| Enhancement                      | Description                                                                                                           |
| -------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **Engagement Uplift Simulation** | Extend comparison page with simulated retention uplift %, and allow user to tune parameters (K, baseline, weighting). |
| **NLP/Google API Integration**   | Enrich features for hybrid modeling.                                                                                  |
| **Caching & Optimization**       | Improve dashboard responsiveness and loading time.                                                                    |

---

## ML Business Case

**Business Objective:**
Estimate the potential uplift in engagement and retention achievable through a predictive recommendation system versus manual selection.

**Proposed ML Approaches (to explore):**

* **Collaborative filtering** using `surprise` or `lightfm`
* **Hybrid model** blending content-based (TF-IDF / embeddings) + behavioral features
* **Gradient boosting models** (`XGBoost`, `CatBoost`) for engagement prediction
* **NLP analysis** *(stretch)* on description (if available) using `nltk` or `spacy`

**Python / ML Libraries (tentative):**
`pandas`, `numpy`, `scikit-learn`, `surprise`, `lightfm`, `xgboost`, `nltk`/`spacy` *(optional)*, `plotly`, `streamlit`

**Evaluation Metrics (offline simulation):**

* **RMSE / R²**:  model accuracy for engagement prediction (BR-2)
* **Simulated Uplift (%)**: increase in mean predicted engagement vs. random and popularity baselines (proxy for retention gain) (BR-3)
* **Diversity Index (entropy)**:  genre fairness across recommendations (BR-4)
* **Precision@K / HitRate@K**: *(optional)*  relevance of top-N predictions

---


### Notebook Structures

```
/notebooks
├── 01_Data_Collection.ipynb      # initial exploration and completeness check of all datasets
├── 01_Data_Cleaning.ipynb        # regex cleaning, NaN handling, and type conversions
├── 02_Data_Imputation.ipynb      # external API enrichment and dictionary-based caching
├── 03_Feature_Engineering.ipynb  # feature creation (engagement_score, recency, popularity)
└── 04_Modeling.ipynb             # clustering and recommendation model development
```

> Each notebook represents a phase of the **CRISP-DM** process:
>
> * **Data Collection (LO7)** – explores and audits dataset quality and structure.
> * **Data Cleaning (LO7.2)** – standardizes and validates fields for analytical use.
> * **Data Imputation (LO7.1)** – retrieves missing values via APIs while ensuring reproducibility.
> * **Feature Engineering (LO4 & LO5)** – generates predictive features tied to business hypotheses.
> * **Modeling (LO5)** – builds and evaluates machine learning pipelines to meet business KPIs.
>
> This modular organization maintains reproducibility and transparency from raw data to model evaluation, meeting the **Code Institute’s CRISP-DM and Learning Outcome** standards for the Predictive Analytics project.

---

## Dashboard Design (Streamlit MVP)

| Page                                                              | Purpose                                                                                              | Key Visuals & Elements                                                                                                     |
| ----------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **1. Executive Summary**                                          | Show KPIs (RMSE, R², engagement score averages).                                                     | Metric cards displaying RMSE/R², mean engagement per strategy, summary text, and simple bar chart for KPI comparison.      |
| **2. Book Analytics Explorer**                                    | Explore correlations and trends.                                                                     | Interactive Plotly scatterplots and heatmaps (e.g. rating vs. genre, review count vs. satisfaction) with dynamic filters. |
| **3. Recommendation Comparison (Model vs. Editorial vs. Random)** | Allow user selection, display predicted scores per list. | User dropdown (simulated user), side-by-side tables of recommendations, and bar chart comparing mean engagement scores.    |
| **4. Insights & Diversity**                                       | Genre representation, fairness, entropy.                                                             | Bar or pie charts showing genre distribution, Shannon entropy indicator, and textual interpretation of diversity trends.   |


---

## MVP vs. Stretch Scope

| Category       | MVP                                                                                                                               | Stretch                                                                                                   |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **Data**       | Use **BBE** and **Goodbooks-10k** datasets for engagement and behavioral signals (8 K overlap subset).                            | Integrate **Google Books API** for additional commercial and descriptive metadata.                        |
| **ML**         | Implement a **baseline regression or collaborative filtering model** to generate predicted engagement scores.                     | Add a **hybrid variant** (e.g., combining regression with genre embeddings or NLP features).              |
| **Visuals**    | Four Streamlit pages: **Executive Summary**, **Analytics Explorer**, **Recommendation Comparison**, and **Insights & Diversity**. | Add an **Uplift Simulator** page with interactive controls (K-slider, baseline selector, uplift metrics). |
| **Evaluation** | Report **RMSE/R²**, **mean engagement comparison** across model vs editorial vs random, and **genre diversity (entropy)**.        | Include **feature-importance plots**, **Precision@K**, or **relevance validation** visualizations.        |
| **Deployment** | Deploy MVP Streamlit app on **Heroku or Streamlit Cloud** (static data + saved model).                                            | Add **caching** and response-time optimization for faster interactivity.                                  |

---


## References

- [Regex101](https://regex101.com/): Online regex tester and debugger.
- [Text Cleaning in Python](https://pbpython.com/text-cleaning.html): A guide on cleaning text data using Python.
- [Pandas Documentation](https://pandas.pydata.org/docs/): datetime, combine_first,
- [NumPy](https://numpy.org/doc/stable/): exponential, logarithm, arange
- [DateUtils Documentation](https://dateutil.readthedocs.io/): for advanced date parsing.
- [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/): for consistent commit messages.
- On Gaussian Distribution: [Free Code Camp](https://www.freecodecamp.org/news/how-to-explain-data-using-gaussian-distribution-and-summary-statistics-with-python/), [Quantinsti](https://blog.quantinsti.com/gaussian-distribution/), [GeeksForGeeks Machine Learning](https://www.geeksforgeeks.org/machine-learning/gaussian-distribution-in-machine-learning/), [eeksForGeeks Python](https://www.geeksforgeeks.org/python/python-normal-distribution-in-statistics/), [PennState College](https://online.stat.psu.edu/stat857/node/77/)
- On Binnin Data: [GeeksForGeeks](https://www.geeksforgeeks.org/numpy/binning-data-in-python-with-scipy-numpy/)
- NotebookLM: Learning guide on data cleaning, to help me find next steps without providing any code.

