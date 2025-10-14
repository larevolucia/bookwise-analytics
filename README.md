
# Bookwise Analytics: Book Club Subscription Optimization

## Business Understanding

### Context

The client is a subscription-based book service. Subscribers pay a monthly fee that grants them one “credit” redeemable for any book in the library.

However, internal data shows a decline in active usage: many subscribers remain subscribed but fail to redeem their monthly credits, eventually leading to churn.

The company currently relies on editorial curation (human staff recommending new titles weekly), which lacks personalization and doesn’t adapt to individual member preferences.

To improve engagement and retention, the business wants to integrate personalized machine learning–driven recommendations alongside its editorial picks.

This project explores how a subscription-based book club can optimize member engagement and retention through data-driven book recommendations.
The aim is to transition from manual, intuition-based book curation to an **evidence-based selection system** that maximizes satisfaction, loyalty, and catalog diversity.

---

### Problem Statement

The business faces a growing inactive user base. While the library continues to expand, users struggle to find content that resonates with their tastes, leading to disengagement and eventual cancellation.

This project aims to move from intuition-based book selection to a data-driven recommendation system that optimizes member satisfaction and long-term retention

---

### Business Objectives

1. **Understand** what book features (e.g. genre, ratings, price, reviews, publication recency) are most correlated with member engagement.
2. **Predict** which upcoming or lesser-known titles are likely to achieve high engagement.
3. **Simulate** potential retention uplift if algorithmic recommendations replace (or supplement) editorial curation.
4. **Ensure** the recommendation system maintains diversity and fairness across genres.

---

### Analytical Goals

* Use cross-platform book data (Goodreads, Goodbooks-10k, Amazon) to **analyze feature correlations** with engagement.
* Build a **hybrid machine learning model** that predicts book satisfaction using historical patterns in ratings and reviews.
* Create a **synthetic retention simulation** to estimate the potential uplift in active user retention when the model is applied.
* Deploy an **interactive Streamlit dashboard** that allows stakeholders to explore insights, prediction outputs, and diversity metrics.

---

### Business Requirements

| ID       | Business Requirement                                                                      | Success Indicator                                  | Dataset(s)             |
| -------- | ----------------------------------------------------------------------------------------- | -------------------------------------------------- | ---------------------- |
| **BR-1** | Identify which book and genre features correlate with higher engagement.                  | Correlation ≥ 0.4 between features and engagement. | BBE, Amazon            |
| **BR-2** | Predict which titles are most likely to achieve high engagement based on historical data. | Model RMSE < 1.0 or R² > 0.7.                      | BBE, Goodbooks, Amazon |
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

This project integrates multiple **publicly available book datasets** to emulate the data ecosystem of a subscription-based digital library. Each dataset contributes a complementary perspective on reader engagement, book attributes, and market behavior.

| Dataset                      | Source                                                                                       | Purpose                                                                         |
| ---------------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| **Best Books Ever** | [GitHub – goodreads_bbe_dataset](https://github.com/scostap/goodreads_bbe_dataset)                             | Core metadata: titles, genres, ratings, popularity (bbeScore, votes).           |
| **Goodbooks-10k**            | [GitHub – goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k)                          | User-book interactions (6M ratings from 53K users) for collaborative filtering. |
| **Amazon Popular Books**     | [GitHub – Amazon-popular-books-dataset](https://github.com/luminati-io/Amazon-popular-books-dataset) | Market context: price, review count, commercial performance.                    |
| **Amazon Bestselling Books**     | [GitHub – Amazon Bestselling Books](https://github.com/suha98/Analysis-of-Amazon-Bestselling-Books-2009-2019) | Market context: price, review count, commercial performance.                    |

> **Why these sources:** Together they capture quality perception (Goodreads), behavioral data (Goodbooks), and commercial success (Amazon); enabling a 360° view of engagement drivers.

Each dataset will be cleaned and unified into a consistent schema *(tentative)* (book_id, title, author, genre, rating, reviews, popularity_score, price, publish_year, user_id)


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

| Goal                                | Visual / Task                                             | Outcome                                 |
| ----------------------------------- | --------------------------------------------------------- | --------------------------------------- |
| Explore data quality & distribution | Histograms, missing-data matrices, pairplots              | Identify cleaning needs & feature gaps  |
| Show engagement vs rating patterns  | Scatter/box by genre; correlation heatmap                 | Quantify relationship strength          |
| Build baseline model                | Popularity baseline + simple CF or regression             | Benchmark for predictive model          |
| Simulate engagement uplift      | Algorithmic uplift vs baselines (random & popularity) | Offline uplift metric |
| Measure diversity                   | Genre entropy & catalog coverage                          | Detect narrow vs broad recommendations  |

### Stretch Goal

| Goal                        | Visual / Task                                 | Outcome                              |
| --------------------------- | --------------------------------------------- | ------------------------------------ |
| Sentiment from reviews | NLP features (TF-IDF / polarity histograms)   | Stronger content-based signals       |


---

## ML Business Case

**Business Objective:**
Estimate the potential uplift in engagement and retention achievable through a predictive recommendation system versus manual selection.

**Proposed ML Approaches (to explore):**

* **Collaborative filtering** using `surprise` or `lightfm`
* **Hybrid model** blending content-based (TF-IDF / embeddings) + behavioral features
* **Gradient boosting models** (`XGBoost`, `CatBoost`) for engagement prediction
* **NLP sentiment analysis** on reviews (if available) using `nltk` or `spacy`

**Python / ML Libraries (tentative):**
`pandas`, `numpy`, `scikit-learn`, `surprise`, `lightfm`, `xgboost`, `nltk`/`spacy` *(optional)*, `plotly`, `streamlit`

**Evaluation Metrics (offline simulation):**

* **RMSE / R²**:  model accuracy for engagement prediction (BR-2)
* **Simulated Uplift (%)**: increase in mean predicted engagement vs. random and popularity baselines (proxy for retention gain) (BR-3)
* **Diversity Index (entropy)**:  genre fairness across recommendations (BR-4)
* **Precision@K / HitRate@K**: *(optional)*  relevance of top-N predictions

---

## Dashboard Design (Streamlit MVP)

| Page                                | Purpose                                                                                           | Key Visuals & Elements                                                                                                |
| ----------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **1. Executive Summary** | Present key KPIs from the modeling stage: RMSE/R², **simulated engagement uplift %**, and genre diversity index. | Metric cards, summary text, simple bar chart comparing Random vs Popularity vs Model strategies. |
| **2. Book Analytics Explorer**      | Explore correlations and patterns between book features (ratings, reviews, genres, year).         | Interactive scatter/heatmap using Plotly; filter by genre or rating.                                                  |
| **3. Insights & Diversity**         | Summarize engagement trends and show genre representation balance.                                | Bar or pie chart for genre shares; entropy indicator; textual interpretation.                                         |
| **4. Uplift Simulator** *(stretch)* | Add interactive controls (K, baseline) to test different scenarios.      | Slider (K), dropdown (baseline), dynamic bar chart for engagement uplift %, with tooltip showing mean predicted engagement (proxy for retention probability). |


---

## MVP vs. Stretch Scope

| Category       | MVP                                                                                               | Stretch                                                                                                  |
| -------------- | ------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Data**       | Use **BBE** and **Goodbooks-10k** datasets for engagement and behavioral signals.                 | Integrate a limited subset of **Amazon Bestselling Books** to analyze publication year or price effects. |
| **ML**         | Implement a **baseline regression or CF model** to predict engagement scores and simulate uplift. | Add a **single hybrid variant** (e.g., regression + genre embeddings) for comparison.                    |
| **Visuals**    | Three Streamlit pages: Executive Summary, Analytics Explorer, and Insights.                       | Add one **Uplift Simulator** page with simple controls (K-slider, baseline selector).                    |
| **Evaluation** | Report **RMSE/R²**, simulated **uplift %**, and **genre diversity (entropy)**.                    | Include a small **feature-importance plot** or **top-K relevance check**.                                |
| **Deployment** | Deploy MVP Streamlit app on Heroku (static data + saved model).                                   | Add lightweight **caching** for faster dashboard loading.                                                |


---
