import os
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ==============================
# Load dataset
df = pd.read_csv(r"mushrooms.csv")

# Clean column names
df.columns = df.columns.str.strip()

# ==============================
# Extract target label and separate features
y = df["class"]

# DROP some columns havibg extremely high correlation with the target variable
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
pickle.dump(X.columns.tolist(), open("model_columns.pkl", "wb"))

# ==============================
# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# ==============================
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
test_data = X_test.copy()
test_data["class"] = y_test

# If we want original labels instead of 0/1:
test_data["class"] = test_data["class"].map({0: "e", 1: "p"})

# Save to CSV
test_data.to_csv("test_data.csv", index=False)

print("test_data.csv saved successfully ✅")
pickle.dump(X_test, open("X_test.pkl", "wb"))
pickle.dump(y_test, open("y_test.pkl", "wb"))

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

# Save the trained models to be used in later predictions
os.makedirs("models", exist_ok=True)

pickle.dump(log_model, open("models/logistic.pkl", "wb"))
pickle.dump(dt_model, open("models/decision_tree.pkl", "wb"))
pickle.dump(knn_model, open("models/knn.pkl", "wb"))
pickle.dump(nb_model, open("models/naive_bayes.pkl", "wb"))
pickle.dump(rf_model, open("models/random_forest.pkl", "wb"))
pickle.dump(xgb_model, open("models/xgboost.pkl", "wb"))

print("Models saved successfully ✅")
