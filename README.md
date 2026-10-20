# Telco Customer Churn Prediction

An end-to-end machine learning project for predicting customer churn using the **Kaggle Playground Series S6E3** dataset.

The project focuses on building a reliable model-selection workflow rather than simply maximizing a leaderboard score. It includes exploratory data analysis, preprocessing, baseline model comparison, progressive hyperparameter tuning, feature-engineering experiments, leakage-safe validation, and final evaluation on both an untouched holdout set and the Kaggle leaderboard.

## Project Overview

The objective is to predict the probability that a customer will churn.

Because the task is evaluated based on the ranking of predicted probabilities, **ROC-AUC** is used as the primary model-selection metric throughout the project.

The modeling workflow follows this structure:

1. Data validation and exploratory data analysis
2. Preprocessing with scikit-learn pipelines
3. Baseline model comparison
4. Hyperparameter tuning
5. Feature engineering and ablation experiments
6. Evaluation on an untouched holdout set
7. Final training on all labeled data
8. Kaggle leaderboard evaluation

---

## Dataset

The project uses the Telco Customer Churn dataset from the **Kaggle Playground Series S6E3** competition.

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

A stratified train/holdout split is created before model selection. The holdout set remains untouched during baseline comparison, hyperparameter tuning, and feature-engineering experiments.

Model selection on the training portion uses **5-fold Stratified Cross-Validation**.

---
## Hyperparameter Tuning

Logistic Regression, Gradient Boosting, and XGBoost were tuned using `GridSearchCV` with ROC-AUC as the optimization metric.

The search was performed progressively, starting with broader parameter ranges and then refining the most promising regions.

Final cross-validation results after tuning:

| Model | Mean CV ROC-AUC | CV Std |
|---|---:|---:|
| **Gradient Boosting** | **0.91657** | 0.00104 |
| XGBoost | 0.91629 | 0.00092 |
| Logistic Regression | 0.90784 | 0.00106 |

Gradient Boosting achieved the highest mean cross-validation ROC-AUC, although its advantage over XGBoost was small.

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

Because these features depend on dataset-level statistics, they were implemented using a custom scikit-learn transformer inside the modeling pipeline. This ensures that the statistics are learned only from the training fold during cross-validation and prevents information leakage.
However, none of the new features really impacted best models, which might be a consequence of the dataset being artificially created from another "original" source, which is not included in this project, even though most of the higher-score projects in the eladerboard have implemented it into their workflow.

## Holdout Evaluation

After model selection and feature-engineering experiments were completed, the selected models were evaluated on the previously untouched holdout set.

Gradient Boosting achieved the highest holdout ROC-AUC, followed closely by XGBoost.

Most importantly, the ranking observed during cross-validation remained consistent on unseen holdout data.

This provided additional confidence that the validation procedure was producing representative estimates of out-of-sample performance.

The tuned **Gradient Boosting model using the original feature set** was selected as the final model.

---

## Final Model

The selected Gradient Boosting model uses:

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

After model selection was completed, the final preprocessing and Gradient Boosting pipeline was retrained on **all available labeled data**.

Predicted churn probabilities were then generated for the Kaggle test set.

---

## Kaggle Results

The final Gradient Boosting model achieved:

| Model | Kaggle Public ROC-AUC | Kaggle Private ROC-AUC |
|---|---:|---:|
| **Gradient Boosting** | **0.91408** | **0.91530** |
| XGBoost | 0.91383 | 0.91506 |

Gradient Boosting slightly outperformed XGBoost on both leaderboard splits.

The Kaggle results were also consistent with the local validation results: Gradient Boosting ranked slightly above XGBoost in cross-validation, on the untouched holdout set, and on both Kaggle leaderboard splits.

---

## Technologies

- Python
- NumPy
- pandas
- Matplotlib
- Seaborn
- scikit-learn
- XGBoost
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

## Running the Project

Clone the repository:

```bash
git clone <repository-url>
cd telco-customer-churn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Download the competition data from Kaggle and update the dataset paths in the notebook if necessary.

Then launch Jupyter:

```bash
jupyter notebook
```

and open:

```text
telco-churn-prediction.ipynb
```

---

## Conclusion

This project demonstrates an end-to-end supervised machine learning workflow with an emphasis on reliable validation and controlled experimentation.

The strongest result was achieved not by maximizing model or feature complexity, but by combining careful preprocessing, systematic hyperparameter tuning, leakage-safe experimentation, and consistent model evaluation.

The final Gradient Boosting model achieved a **0.91530 private Kaggle ROC-AUC**.