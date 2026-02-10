import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Heart Disease Classification", layout="wide")

st.title("Heart Disease Classification Models")
st.write("Upload the dataset and evaluate different ML models")

# Sidebar: File upload
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload heart disease CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(data.head())
else:
    st.info("Please upload the heart disease CSV file to proceed.")
    st.stop()



st.subheader("Preprocessing")

# --- Binary encode sex ---
data['sex'] = data['sex'].map({'Male': 1, 'Female': 0})

# --- One-hot encode categorical columns ---
cat_cols = data.select_dtypes(include='object').columns
data = pd.get_dummies(data, columns=cat_cols, drop_first=True)

# --- Convert bool to int ---
bool_cols = data.select_dtypes(include='bool').columns
data[bool_cols] = data[bool_cols].astype(int)

# --- Create binary target ---
data['target'] = data['num'].apply(lambda x: 1 if x >= 1 else 0)
data.drop(columns=['num'], inplace=True)

st.success("Preprocessing completed successfully!")

with st.expander("Show preprocessed data"):
    st.write("Shape after preprocessing:", data.shape)
    st.dataframe(data.head())


    st.subheader("Train–Test Split")

# Separate features and target
X = data.drop(columns=['target'])
y = data['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

st.write("Training samples:", X_train.shape[0])
st.write("Testing samples:", X_test.shape[0])

st.subheader("Model Selection")

model_name = st.selectbox(
    "Choose a classification model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from xgboost import XGBClassifier

# Scaling (only for required models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if model_name == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

elif model_name == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

elif model_name == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

elif model_name == "Naive Bayes":
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

elif model_name == "Random Forest":
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

elif model_name == "XGBoost":
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]


    st.subheader("Model Performance")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

st.write(f"**Accuracy:** {accuracy:.4f}")
st.write(f"**Precision:** {precision:.4f}")
st.write(f"**Recall:** {recall:.4f}")
st.write(f"**F1 Score:** {f1:.4f}")
st.write(f"**AUC:** {auc:.4f}")

cm = confusion_matrix(y_test, y_pred)
st.write("**Confusion Matrix:**")
st.dataframe(cm)

