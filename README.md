# Employee Attrition Prediction using SVM

## Objective
This project aims to predict whether an employee will leave the company (attrition) using Support Vector Machine (SVM) models. Due to class imbalance in the dataset, handling this imbalance became a key focus area to improve model generalization, especially for predicting minority class outcomes.

---

## Dataset
The dataset includes various employee attributes such as:
- Demographics (e.g., Age, Gender, MaritalStatus)
- Job-related data (e.g., JobRole, MonthlyIncome, JobSatisfaction)
- Performance metrics and years of experience

The target variable is:
- `Attrition`: Whether an employee left the company or not (`Yes` / `No`)

The dataset used in this project is available on Kaggle:  
[Employee Attrition Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## Methods

### Preprocessing
- One-hot encoding was applied to categorical variables.
- The target variable was separated as `y`, while all other columns were kept as features `X`.

### Imbalance Handling
- The dataset was **highly imbalanced**, with a majority of 'No Attrition' cases.
- To address this, **SMOTE (Synthetic Minority Oversampling Technique)** was used, a new tool applied in this project to create synthetic samples of the minority class.

### Model Training
- Model: **Support Vector Machine (SVM)**
- Validation Strategy: **Stratified K-Fold Cross-Validation** (k=5)
- Evaluation Metrics per fold:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - Confusion Matrix

### Evaluation & Visualization
- A **learning curve** was plotted to visualize training vs testing performance.
- A **ROC curve** was plotted to measure classifier performance.
- A Grid search was applied for choosing the best parameters

---

## Observations & Analysis

- The model performed well overall, with decent average precision and F1-score.
- **Slight overfitting** was observed in the learning curve — training performance was higher than test performance in some folds.
- This overfitting is **likely due to the data imbalance** despite applying SMOTE.
- Class 1 (attrition = "Yes") was harder to classify correctly.
- Applying SMOTE improved **recall on the minority class**, but required careful tuning of the SVM to maintain balance across metrics.

---

## Notes
- Future work could explore ensemble methods (e.g., Random Forest, XGBoost) or cost-sensitive learning to further improve minority class predictions.
