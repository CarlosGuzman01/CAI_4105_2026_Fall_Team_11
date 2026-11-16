# CAI 4105 – Team 11

## Members

1. Abhiram Vedipala
2. Nicoll Verjan
3. Carlos Guzman

## Challenge 2 - Classification Problem: Online Purchase Prediction

**Notebook**: `2-challenge/final_notebook.ipynb` is the final notebook. It runs the full workflow from loading the data to writing `predictions.csv`. `2-challenge/notebook.ipynb` is the earlier data-loading starter.

**Other files**: `2-challenge/predictions.csv` contains `ID` and the predicted `Revenue` (True/False) for all 3,083 test sessions. `2-challenge/data/` contains the provided CSVs. `2-challenge/environment.yml` pins the versions used to run the notebook.

**Environment (Anaconda)**: from inside `2-challenge/`:

```
conda env create -f environment.yml
conda activate cai4105-team11-c2
python -m ipykernel install --user --name cai4105-team11-c2
jupyter lab final_notebook.ipynb
```

**Reproduce**: select the `cai4105-team11-c2` kernel and run the notebook top to bottom from inside `2-challenge/` (about 2 minutes on a laptop, mostly grid search). The seed is fixed at 42, and re-running gives the same `predictions.csv`.

**Continuity with Challenge 1**: we reuse the Challenge 1 workflow: seed 42, 5-fold CV on training data only, fold-safe sklearn Pipelines, an `evaluate()` helper that reports train vs CV scores to diagnose overfitting, the required models plus a HistGradientBoosting extra, Grid Search on the strongest models, and a refit of the final model on all training data. Challenge 2 changes what the problem requires: stratified folds and positive-class F1 instead of RMSE (the target is imbalanced), and a decision threshold chosen on out-of-fold predictions.

**Approach**: `ID` is dropped. `Month`, `VisitorType`, `OperatingSystems`, `Browser`, `Region`, and `TrafficType` are one-hot encoded, ignoring unseen categories. Skewed page-count, duration, and `PageValues` features are log-transformed; numeric features are scaled for the linear, KNN, and SVM models. Engineered features: `TotalPages`, `TotalDuration`, `HasPageValue` (`PageValues > 0`), `ProductDurationPerPage`, and `IsQ4`. All preprocessing lives inside sklearn Pipelines, so nothing leaks across folds.

**Validation**: stratified 5-fold CV on the training data only (seed 42), with **F1 for `Revenue = True`** as the primary metric. Precision, recall, ROC-AUC, and average precision support it, and train-fold F1 is used to diagnose overfitting. The same folds are used for every comparison and every Grid Search. The decision threshold is chosen on out-of-fold training predictions. The test set is used only for the final predictions. About 15.5% of training sessions are purchases, so accuracy is not used to compare models.

**Models evaluated**: all seven required models (baseline CV F1), then Grid Search on five of them.

| Model | Baseline CV F1 | Tuned CV F1 | Train F1 (tuned) |
|---|---:|---:|---:|
| **Random Forest (final)** | 0.660 | **0.687** | 0.823 |
| Gradient Boosting (HistGB when tuned) | 0.664 | 0.673 | 0.823 |
| Support Vector Machine (RBF) | 0.663 | 0.672 | 0.678 |
| Logistic Regression | 0.658 | 0.671 | 0.674 |
| K-Nearest Neighbors | 0.645 | 0.654 | 0.677 |
| Decision Tree | 0.637 | – | – |
| Naive Bayes (Gaussian) | 0.302 | – | – |

**Final model**: Random Forest (300 trees, `max_depth=12`, `min_samples_leaf=1`, `class_weight="balanced_subsample"`) with a probability threshold of **0.552**, refit on all training data. **Out-of-fold F1 = 0.693** (precision 0.644, recall 0.750, ROC-AUC 0.930). It has the highest tuned F1, and depth limiting reduced the untuned forest's overfitting gap from 0.34 to 0.14. It trains in seconds and exposes feature importances; Logistic Regression is the simpler fallback (CV F1 0.671).


