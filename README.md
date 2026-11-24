# Telco Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the **Kaggle Playground Series S6E3** dataset.

The project focuses on building a reliable model-selection workflow rather than simply maximizing a leaderboard score. It includes exploratory data analysis, preprocessing, baseline model comparison, hyperparameter optimization, feature-engineering experiments, leakage-safe validation, OOF ensemble blending, and final evaluation on both a holdout set and the Kaggle leaderboard.

## Project Overview

The objective is to predict the probability that a customer will churn.

Because the task is evaluated based on the ranking of predicted probabilities, **ROC-AUC** is used as the primary model-selection metric throughout the project.

The modeling workflow follows this structure:

1. Data validation and exploratory data analysis
2. Preprocessing with scikit-learn pipelines
3. Baseline model comparison
4. Feature engineering and ablation experiments
5. Hyperparameter optimization with Optuna
6. Cross-validation and out-of-fold (OOF) predictions
7. Ensemble blending with automated weight optimization
8. Holdout evaluation
9. Final training on all labeled data
10. Kaggle leaderboard evaluation

---

## Dataset

The project uses the Telco Customer Churn dataset from the [Kaggle Playground Series S6E3](https://www.kaggle.com/competitions/playground-series-s6e3) competition.

The data contains customer-level information such as:

- tenure;
- contract type;
- internet service;
- monthly and total charges;
- online security and technical support;
- streaming services;
- payment method;
- demographic and account-related attributes.

The target variable is `Churn`.

The Kaggle competition files are not included in this repository and should be downloaded separately from Kaggle.

---

## Preprocessing

Features are separated into numerical, ordinal/binary, and nominal categorical groups.

Preprocessing is implemented with scikit-learn `Pipeline` and `ColumnTransformer` objects so that transformations are applied consistently during cross-validation and final training.

The preprocessing workflow includes:

- standardization of numerical features;
- ordinal encoding of binary and ordered categorical variables;
- one-hot encoding of nominal categorical variables.

A stratified train/holdout split is created before model selection.

Model selection on the training portion uses **5-fold Stratified Cross-Validation**.

---

## Feature Engineering

Several groups of engineered features were evaluated through controlled ablation experiments.

### Row-wise features

The experiments included:

- arithmetic relationships between tenure and charges;
- number of subscribed services;
- categorical interaction features;
- tenure-based features;
- squared numerical features;
- logarithmic transformations.

### Context-dependent features

Additional features were created using statistics learned from the training data, including:

- quantile-based bins;
- category frequencies;
- group-level monthly charge averages.

Because these features depend on dataset-level statistics, they were implemented using a custom scikit-learn transformer inside the modeling pipeline. This ensures that statistics are learned only from the training fold during cross-validation, preventing data leakage.

Statistical tests, Holm correction, and correlation analysis were used to support the interpretation of experimental results.

Additional engineered features did not consistently improve model performance. The strongest final models retained the original feature set.

---

## Hyperparameter Tuning

Logistic Regression, Gradient Boosting, and XGBoost were optimized using **Optuna** with ROC-AUC as the optimization metric.

The optimization was performed on a sample of 100,000 observations using 5-fold stratified cross-validation:

- Logistic Regression: 20 trials;
- Gradient Boosting: 20 trials;
- XGBoost: 40 trials.

The selected configurations were subsequently evaluated using cross-validation on the full training split.

| Model | Mean CV ROC-AUC |
|---|---:|
| **Original Gradient Boosting** | **0.916572** |
| Optuna XGBoost | 0.915920 |

Although XGBoost achieved competitive performance, the original Gradient Boosting benchmark remained the strongest standalone model.

---

## Out-of-Fold Ensemble

The original Gradient Boosting benchmark and Optuna-tuned XGBoost were combined using weighted probability averaging.

OOF predictions were generated using 5-fold stratified cross-validation. Ensemble weights were optimized on 70% of the OOF predictions and evaluated on the remaining 30%.

The optimized weights were:

- **Gradient Boosting: 79%**
- **XGBoost: 21%**

| Model | OOF Validation ROC-AUC |
|---|---:|
| Gradient Boosting | 0.917151 |
| XGBoost | 0.916441 |
| **Ensemble** | **0.917184** |

The ensemble marginally outperformed both individual models on the OOF validation subset.

---

## Holdout Evaluation

The individual models and the weighted ensemble were evaluated on the holdout set.

| Model | Holdout ROC-AUC |
|---|---:|
| **Gradient Boosting** | **0.917269** |
| Ensemble | 0.917260 |
| XGBoost | 0.916409 |

Gradient Boosting slightly outperformed the ensemble on the holdout set. However, the difference was negligible, indicating comparable predictive performance.

---

## Final Model

The final submission uses a weighted ensemble of **Gradient Boosting and XGBoost**, both trained on the original feature set.

The Gradient Boosting configuration is:

```python
GradientBoostingClassifier(
    random_state=42,
    learning_rate=0.1,
    max_depth=4,
    n_estimators=500,
    min_samples_leaf=10,
    subsample=0.85
)
```

Both models were retrained on all available labeled data.

The final churn probabilities were calculated using the OOF-optimized weights:

```python
final_pred = 0.79 * gb_pred + 0.21 * xgb_pred
```

The resulting predictions were submitted to Kaggle.

---

## Kaggle Results

| Model | Kaggle Public ROC-AUC | Kaggle Private ROC-AUC |
|---|---:|---:|
| Gradient Boosting | 0.91408 | **0.91530** |
| **Final Ensemble** | **0.91412** | **0.91530** |

The ensemble achieved a marginal improvement on the public leaderboard and matched the standalone Gradient Boosting model on the private leaderboard.

The results suggest that both approaches provide comparable predictive performance, without establishing a meaningful advantage for the more complex ensemble.

---

## Technologies

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- SciPy
- Statsmodels
- scikit-learn
- XGBoost
- Optuna
- Jupyter Notebook

---

## Repository Structure

```text
telco-customer-churn/
│
├── README.md
├── telco-churn-prediction.ipynb
├── requirements.txt
├── .gitignore
│
└── data/
    └── README.md
```

The main analysis, experiments, and modeling workflow are contained in `telco-churn-prediction.ipynb`.

Raw Kaggle competition data is intentionally not stored in the repository.

---

## Conclusion

This project demonstrates an end-to-end supervised machine learning workflow with an emphasis on reliable validation, controlled experimentation, hyperparameter optimization, and ensemble learning.

The results highlight several important findings:

- Gradient Boosting remained the strongest standalone model.
- Additional feature engineering did not consistently improve performance.
- Optuna optimization did not outperform the original Gradient Boosting benchmark.
- OOF ensemble blending provided a marginal improvement during validation.
- Increased model complexity did not result in a meaningful improvement in holdout or leaderboard performance.

The final ensemble achieved a **0.91530 private ROC-AUC** and a **0.91412 public ROC-AUC** on Kaggle.
