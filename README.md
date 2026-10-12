# CAI 4105 – Team 11

## Members

1. Abhiram Vedipala
2. Nicoll Verjan
3. Carlos Guzman

## Challenge 1 – Apartment Rental Price Prediction (regression)

**Files (in `1-challenge/`)**: `notebook.ipynb` (full workflow), `predictions.csv` (ID + predicted price for all 2,467 test rows), `data/` (provided CSVs), `requirements.txt`.

**Reproduce**: `pip install -r requirements.txt`, then run `1-challenge/notebook.ipynb` top to bottom from inside `1-challenge/` (about 15 min, almost all of it grid search). Seed is fixed at 42.

**Approach**: `ID` and the near-constant `category` are dropped; `has_photo` and `state` are one-hot encoded (unknown states ignored, since `WY` only appears in test); numerics are scaled only for linear models. All preprocessing and feature engineering (`log_sqft`, `total_rooms`, `sqft_per_bed`, `bath_per_bed`) live inside sklearn Pipelines, so nothing leaks across folds. Models are trained on `log1p(price)` and predictions are inverted, with RMSE always measured in dollars.

**Validation**: 5-fold shuffled KFold on the training data only (RMSE primary metric), used for every comparison and every Grid Search. Train vs CV RMSE is used to diagnose over/underfitting. The test set is only used for the final predictions.

**Models evaluated**: Linear Regression, Decision Tree, Random Forest, Gradient Boosting (required), plus Ridge and HistGradientBoosting. Mean-predictor baseline CV RMSE is about $821.

| Model (best after tuning) | Train RMSE | CV RMSE |
|---|---|---|
| **Gradient Boosting (final)** | 191 | **390.7** |
| Hist Gradient Boosting | 262 | 396.4 |
| Random Forest | 177 | 420.9 |
| Decision Tree | 345 | 487.3 |
| Ridge | 507 | 512.3 |
| Linear Regression (untuned) | 521 | 526.8 |

**Final model**: tuned Gradient Boosting (GridSearchCV over n_estimators, learning_rate, max_depth, subsample), refit on all training data. Lowest CV RMSE (~$391), moderate train/CV gap, trains in seconds. Less interpretable than linear regression, but permutation importance shows sensible drivers (location, square footage, bathrooms).
