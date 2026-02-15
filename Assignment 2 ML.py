import pandas as pd
import numpy as np

# ==============================
# Load dataset
df = pd.read_csv(r"C:\Users\user\Desktop\mushrooms.csv")

# Clean column names
df.columns = df.columns.str.strip()

# ==============================
# Separate target label
y = df["class"]

# DROP some columns you chose earlier
X = df.drop(columns=["class", "odor", "gill-color", "veil-type"])

# One-hot encode categorical features
X = pd.get_dummies(X, drop_first=True)

# ==============================
# CORRELATION-BASED FEATURE SELECTION
# ==============================

# Compute absolute correlation matrix
corr_matrix = X.corr().abs()

# Create upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Define threshold
threshold = 0.75

# Identify columns to drop if they have correlation > threshold
to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

print("Dropping highly correlated features:", to_drop)

# Drop those features
X = X.drop(columns=to_drop)

# ==============================
# Encode target labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# ==============================
# Train-Test Split + Scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==============================
# TRAIN MODELS
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

log_model = LogisticRegression(max_iter=2000)
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier()
nb_model = GaussianNB()
rf_model = RandomForestClassifier()
xgb_model = XGBClassifier(eval_metric="mlogloss")

models = {
    "Logistic Regression": log_model,
    "Decision Tree": dt_model,
    "KNN": knn_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model,
}

for name, model in models.items():
    model.fit(X_train, y_train)

print("All models trained successfully with correlation-selected features ✅")

# ==============================
# EVALUATE MODELS
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

results = []

for name, model in models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob[:, 1])
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append([name, accuracy, precision, auc, recall, f1, mcc])

results_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "AUC", "Recall", "F1 Score", "MCC"],
)

print("\nModel Comparison After Correlation Selection:\n")
print(results_df)


import os
import pickle

os.makedirs("models", exist_ok=True)

pickle.dump(log_model, open("models/logistic.pkl", "wb"))
pickle.dump(dt_model, open("models/decision_tree.pkl", "wb"))
pickle.dump(knn_model, open("models/knn.pkl", "wb"))
pickle.dump(nb_model, open("models/naive_bayes.pkl", "wb"))
pickle.dump(rf_model, open("models/random_forest.pkl", "wb"))
pickle.dump(xgb_model, open("models/xgboost.pkl", "wb"))

print("Models saved successfully ✅")



import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 E-Commerce Category Prediction App")

# ============================
# Upload Dataset
# ============================
uploaded_file = st.file_uploader("Upload CSV", type="csv")

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # ============================
    # Preprocessing (Same as Training)
    # ============================

    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Day"] = df["Order Date"].dt.day
    df = df.drop(columns=["Order Date"])

    y = df["Category"]
    X = df.drop(columns=["Category", "Product Name"])

    X = pd.get_dummies(X, drop_first=True)

    le = LabelEncoder()
    y = le.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ============================
    # Model Selection
    # ============================
    model_choice = st.selectbox(
        "Select Model",
        ["Logistic Regression", "Decision Tree", "KNN",
         "Naive Bayes", "Random Forest", "XGBoost"]
    )

    if model_choice == "Logistic Regression":
        model = pickle.load(open("models/logistic.pkl", "rb"))
    elif model_choice == "Decision Tree":
        model = pickle.load(open("models/decision_tree.pkl", "rb"))
    elif model_choice == "KNN":
        model = pickle.load(open("models/knn.pkl", "rb"))
    elif model_choice == "Naive Bayes":
        model = pickle.load(open("models/naive_bayes.pkl", "rb"))
    elif model_choice == "Random Forest":
        model = pickle.load(open("models/random_forest.pkl", "rb"))
    else:
        model = pickle.load(open("models/xgboost.pkl", "rb"))

    # ============================
    # Predictions
    # ============================
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)

    st.subheader("📈 Model Performance")
    st.write(f"Accuracy: {accuracy:.4f}")

    # ============================
    # Confusion Matrix
    # ============================
    st.subheader("📊 Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    st.pyplot(fig)

