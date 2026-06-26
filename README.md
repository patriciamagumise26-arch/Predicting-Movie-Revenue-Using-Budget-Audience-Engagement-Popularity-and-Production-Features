#  Predicting Movie Revenue Using Budget, Audience Engagement, Popularity, and Production Features

A regression project that predicts box-office revenue from pre-release and production metadata budget, audience engagement signals, popularity, and production attributes  using the TMDB movie dataset.

##  Project Overview

Studios, distributors, and investors need to make marketing, release-timing, and investment decisions long before a film's true performance is known. This project builds and compares several regression models to predict movie revenue from features available pre- or at-release, and identifies which factors actually drive box-office outcomes.

**Problem type:** Regression (revenue is continuous)
**Dataset:** [TMDB Movie Dataset](https://www.kaggle.com/) (Kaggle) — budget, popularity, vote average/count, runtime, language, genres, production companies/countries, release date, etc.
**Final dataset size:** 3,376 rows × 15 columns after cleaning

##  Key Findings

- **Vote count (r = 0.76)**, **budget (r = 0.71)**, and **popularity (r = 0.61)** are by far the strongest predictors of revenue — the only three features correlated above 0.5 with the target.
- Revenue is heavily right-skewed (≈49% of movies earn under $50M, <1% exceed $1B), so the target was log-transformed using a `TransformedTargetRegressor`.
- Because the top predictors are correlated with each other (e.g., popularity vs. vote count, r = 0.75), tree-based models that can capture non-linear interactions substantially outperform linear models.
- **Random Forest was the best model overall**, achieving the lowest test RMSE ($114M) and the most stable cross-validation performance ($3.8M std).

##  Methodology

**1. Data Cleaning & Preprocessing**
- Dropped low-value columns (`homepage`, `tagline`, `id`, `title`, `overview`, `original_title`)
- Removed duplicates and rows with missing/zero revenue
- Parsed nested JSON-like fields (genres, production companies/countries, spoken languages, keywords) into numeric count features
- Engineered a cyclic (sine/cosine) `release_month` feature from release date
- Built a `ColumnTransformer` pipeline routing numeric, categorical, text (TF-IDF on genres), and cyclic-time features through dedicated branches
- Median imputation for numeric features, most-frequent imputation for categorical features
- Standard scaling for numeric features; one-hot encoding for categorical features
- 80/10/10 train/validation/test split (`random_state=42`)

**2. Modeling**
Six models were trained and tuned via `GridSearchCV`, all wrapped in a `TransformedTargetRegressor` (log1p / expm1) to handle the skewed target:
- Ridge Regression
- Decision Tree
- Support Vector Regression (SVR)
- Random Forest
- Gradient Boosting
- Neural Network (MLP)

**3. Evaluation**
RMSE (in dollars) was used as the primary metric since it penalizes large errors heavily appropriate when blockbuster-scale prediction errors matter most.

| Model | Test RMSE | CV Std. Dev. |
|---|---|---|
| **Random Forest** | **$114M** | **$3.8M** |
| Neural Network | $122M | N/A |
| Gradient Boosting | $123M | N/A |
| SVR | $133M | $6.5M |
| Decision Tree | $137M | $5.8M |
| Ridge Regression | $1.57B | very high |

##  Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `matplotlib` · `seaborn`


##  Author

**Patricia Magumise**
