# Heart Disease Classification – ML Assignment 2

## Problem Statement

The objective of this assignment is to implement and compare multiple machine learning classification models to predict the presence of heart disease using clinical data. The models are evaluated using standard classification metrics and deployed using Streamlit Community Cloud.

---

## Dataset Description

The dataset used is a Heart Disease dataset obtained from Kaggle. It contains more than 500 instances and more than 12 features including:

- Age
- Sex
- Chest pain type
- Cholesterol level
- Resting blood pressure
- ECG results
- Thalassemia
- Exercise-induced angina
- Maximum heart rate

The original target variable was converted into a binary classification problem:

- 0 → No heart disease
- 1 → Presence of heart disease

---

## Models Used

The following classification models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest (Ensemble)
6. XGBoost (Ensemble)

---

## Model Comparison Table

| ML Model            | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
| ------------------- | -------- | ----- | --------- | ------ | -------- | ----- |
| Logistic Regression | 0.924    | 0.950 | 0.923     | 0.941  | 0.932    | 0.846 |
| Decision Tree       | 0.880    | 0.875 | 0.870     | 0.922  | 0.895    | 0.758 |
| KNN                 | 0.875    | 0.935 | 0.883     | 0.892  | 0.888    | 0.747 |
| Naive Bayes         | 0.864    | 0.907 | 0.881     | 0.873  | 0.877    | 0.725 |
| Random Forest       | 0.897    | 0.968 | 0.888     | 0.931  | 0.909    | 0.791 |
| XGBoost             | 0.918    | 0.964 | 0.914     | 0.941  | 0.928    | 0.835 |

---

## Model Observations

| ML Model            | Observation about model performance                                                              |
| ------------------- | ------------------------------------------------------------------------------------------------ |
| Logistic Regression | Provided strong overall performance with high recall and AUC. Balanced and interpretable model.  |
| Decision Tree       | Good recall but lower overall stability. Shows signs of overfitting compared to ensemble models. |
| KNN                 | Moderate performance. Sensitive to feature scaling and dataset size.                             |
| Naive Bayes         | Comparatively lower performance due to independence assumptions.                                 |
| Random Forest       | Strong ensemble performance with improved stability and higher AUC.                              |
| XGBoost             | High performance with strong recall and AUC. Boosting improves classification robustness.        |

---

## Deployment

Live Streamlit App:
https://heart-disease-ml-assignmentgit-ceoojoarnggfpxzlqxe2nh.streamlit.app/

GitHub Repository:
https://github.com/prakarti/heart-disease-ml-assignment

## Files Included in Repository

- app.py – Streamlit application
- 2025AA05584.ipynb – Model training and evaluation code
- heart_disease_cleaned.csv – Test dataset
- requirements.txt – Required libraries
