import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    f1_score
)

# ===============================
# Page Configuration
# ===============================
st.set_page_config(
    page_title="Mushroom Classification Dashboard",
    page_icon="🍄",
    layout="wide"
)

st.title("🍄 Mushroom Classification - Model Comparison Dashboard")

# ==========================================================
# 1️⃣ Correlation Analysis Section
# ==========================================================
st.header("📊 Exploratory Data Analysis - Correlation")

# Load dataset
df = pd.read_csv("mushrooms.csv")

# Convert target to numeric
df["class"] = df["class"].map({"e": 0, "p": 1})

# One-hot encode
df_encoded = pd.get_dummies(df)

# Correlation with target
corr_target = df_encoded.corr()["class"].sort_values(ascending=False)

# Top 10 positively correlated features (excluding target itself)
top_corr = corr_target[1:11]

plt.figure(figsize=(8, 6))
plt.barh(top_corr.index, top_corr.values)
plt.gca().invert_yaxis()
plt.title("Top 10 Features Correlated with Poisonous Class")
plt.xlabel("Correlation Value")
st.pyplot(plt)

st.markdown(
"""
**Interpretation:**  
Features with higher correlation values are more strongly associated with the mushroom being poisonous.
"""
)

# ==========================================================
# 2️⃣ Load Saved Models
# ==========================================================
logistic_model = pickle.load(open("logistic_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
X_test = pickle.load(open("X_test.pkl", "rb"))
y_test = pickle.load(open("y_test.pkl", "rb"))

# ==========================================================
# 3️⃣ Model Predictions
# ==========================================================
log_pred = logistic_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

log_prob = logistic_model.predict_proba(X_test)[:, 1]
rf_prob = rf_model.predict_proba(X_test)[:, 1]

# ==========================================================
# 4️⃣ Performance Metrics
# ==========================================================
st.header("📈 Model Performance Comparison")

log_acc = accuracy_score(y_test, log_pred)
rf_acc = accuracy_score(y_test, rf_pred)

log_f1 = f1_score(y_test, log_pred, pos_label="p")
rf_f1 = f1_score(y_test, rf_pred, pos_label="p")

col1, col2 = st.columns(2)

with col1:
    plt.figure()
    plt.bar(["Logistic", "Random Forest"], [log_acc, rf_acc])
    plt.ylim(0, 1)
    plt.title("Accuracy Comparison")
    plt.ylabel("Score")
    st.pyplot(plt)

with col2:
    plt.figure()
    plt.bar(["Logistic", "Random Forest"], [log_f1, rf_f1])
    plt.ylim(0, 1)
    plt.title("F1 Score (Poisonous Class)")
    plt.ylabel("Score")
    st.pyplot(plt)

# ==========================================================
# 5️⃣ ROC Curve
# ==========================================================
st.header("📊 ROC Curve")

# Convert y_test to numeric for ROC
y_test_numeric = np.where(y_test == "p", 1, 0)

fpr_log, tpr_log, _ = roc_curve(y_test_numeric, log_prob)
fpr_rf, tpr_rf, _ = roc_curve(y_test_numeric, rf_prob)

roc_auc_log = auc(fpr_log, tpr_log)
roc_auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_log, tpr_log, label=f"Logistic (AUC = {roc_auc_log:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
st.pyplot(plt)

# ==========================================================
# 6️⃣ Confusion Matrix
# ==========================================================
st.header("🔎 Confusion Matrix - Random Forest")

cm = confusion_matrix(y_test, rf_pred)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.xticks([0, 1], ["Edible", "Poisonous"])
plt.yticks([0, 1], ["Edible", "Poisonous"])
plt.colorbar()
st.pyplot(plt)

# ==========================================================
# 7️⃣ Feature Importance
# ==========================================================
st.header("⭐ Top 10 Feature Importance - Random Forest")

importances = rf_model.feature_importances_
feature_names = encoder.get_feature_names_out()

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

plt.figure()
plt.barh(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.xlabel("Importance Score")
st.pyplot(plt)

# ==========================================================
# 8️⃣ Interactive Prediction
# ==========================================================
st.header("🧪 Try Your Own Prediction")

cap_shape = st.selectbox("Cap Shape", ['b','c','x','f','k','s'])
cap_surface = st.selectbox("Cap Surface", ['f','g','y','s'])
cap_color = st.selectbox("Cap Color", ['n','b','c','g','r','p','u','e','w','y'])
bruises = st.selectbox("Bruises", ['t','f'])
gill_size = st.selectbox("Gill Size", ['b','n'])

if st.button("Predict Mushroom Type"):

    input_df = pd.DataFrame([[ 
        cap_shape,
        cap_surface,
        cap_color,
        bruises,
        gill_size
    ]], columns=[
        "cap-shape",
        "cap-surface",
        "cap-color",
        "bruises",
        "gill-size"
    ])

    encoded_input = encoder.transform(input_df)
    prediction = rf_model.predict(encoded_input)

    if prediction[0] == "p":
        st.error("⚠️ Poisonous Mushroom")
    else:
        st.success("✅ Edible Mushroom")
