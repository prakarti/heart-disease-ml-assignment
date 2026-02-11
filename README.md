# Heart Disease Classification – ML Assignment 2

This project implements multiple machine learning models to classify the presence of heart disease using a structured dataset. The application is deployed using Streamlit.

## Dataset

The dataset contains clinical and medical attributes such as:

- Age
- Sex
- Chest pain type
- Cholesterol level
- Blood pressure
- ECG results
- Thalassemia
- etc.

The original target column was converted into a binary classification problem:

- 0 → No heart disease
- 1 → Presence of heart disease

## Preprocessing Steps

- Binary encoding for gender
- One-hot encoding for categorical features
- Conversion of boolean values to integers
- Train-test split (80-20)
- Feature scaling where required

## Models Implemented

The following models were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest
6. XGBoost

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- AUC Score
- Confusion Matrix

## Deployment

The application is deployed using Streamlit Cloud.

Live App:
https://heart-disease-ml-assignmentgit-ceoojoarnggfpxzlqxe2nh.streamlit.app/

## How to Run Locally

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the app:
   streamlit run app.py

## Conclusion

Logistic Regression and XGBoost performed best overall based on recall and AUC score.  
Random Forest also showed strong performance due to ensemble learning.

Naive Bayes showed comparatively lower performance.  
Decision Tree showed signs of overfitting compared to ensemble models.
