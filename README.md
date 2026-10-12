# CAI 4105 – Team 11

## Members

1. Abhiram Vedipala
2. Nicoll Verjan
3. Carlos Guzman

## Challenge 1 – Apartment Rental Price Prediction (regression)

**Notebooks**: `1-challenge/final_notebook.ipynb` is the final notebook for the assignment submission. The notebooks in `1-challenge/other-notebooks/` were used during analysis and development; they are supporting work only and are not the notebook to submit.

**Other files**: `1-challenge/predictions.csv` contains ID + predicted price for all 2,467 test rows; `1-challenge/data/` contains the provided CSVs; `1-challenge/requirements.txt` lists dependencies.

**Environment (Anaconda)**: we use an Anaconda/conda environment, defined in `1-challenge/environment.yml` (Python 3.13, scikit-learn 1.9.1, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2, Jupyter). Setup, from inside `1-challenge/`:

```
conda env create -f environment.yml
conda activate cai4105-team11
python -m ipykernel install --user --name cai4105-team11
jupyter notebook final_notebook.ipynb
```

(`requirements.txt` lists the same versions for plain pip users.)

**Reproduce**: select the `cai4105-team11` kernel and run `1-challenge/final_notebook.ipynb` top to bottom from inside `1-challenge/` (about 15 min, almost all of it grid search). Seed is fixed at 42. Re-running in this environment gave a final Gradient Boosting CV RMSE of about 391, matching the reported 390.7 within rounding.

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
