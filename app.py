import os
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

# Necessary initializations
model_columns = pickle.load(open("model_columns.pkl", "rb"))
df_full = pd.read_csv("mushrooms.csv")
feature_columns = df_full.drop("class", axis=1).columns.tolist()

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
models_path = os.path.join("models", "{model}")

logistic_model = pickle.load(
    open(models_path.format(model="logistic.pkl"), "rb")
)
rf_model = pickle.load(
    open(models_path.format(model="random_forest.pkl"), "rb")
)
decision_tree_model = pickle.load(
    open(models_path.format(model="decision_tree.pkl"), "rb")
)
naive_bayes_model = pickle.load(
    open(models_path.format(model="naive_bayes.pkl"), "rb")
)
knn_model = pickle.load(
    open(models_path.format(model="knn.pkl"), "rb")
)
xgboost_model = pickle.load(
    open(models_path.format(model="xgboost.pkl"), "rb")
)


X_test = pickle.load(open("X_test.pkl", "rb"))
y_test = pickle.load(open("y_test.pkl", "rb"))

# ==========================================================
# 3️⃣ Model Predictions
# ==========================================================
log_pred = logistic_model.predict(X_test)
rf_pred = rf_model.predict(X_test)
dt_pred = decision_tree_model.predict(X_test)
nb_pred = naive_bayes_model.predict(X_test)
knn_pred = knn_model.predict(X_test)
xg_pred = xgboost_model.predict(X_test)

log_prob = logistic_model.predict_proba(X_test)[:, 1]
rf_prob = rf_model.predict_proba(X_test)[:, 1]
dt_prob = decision_tree_model.predict_proba(X_test)[:, 1]
nb_prob = naive_bayes_model.predict_proba(X_test)[:, 1]
knn_prob = knn_model.predict_proba(X_test)[:, 1]
xg_prob = xgboost_model.predict_proba(X_test)[:, 1]


# ==========================================================
# 4️⃣ Performance Metrics
# ==========================================================
st.header("📈 Model Performance Comparison")

log_acc = accuracy_score(y_test, log_pred)
rf_acc = accuracy_score(y_test, rf_pred)
dt_acc = accuracy_score(y_test, dt_pred)
nb_acc = accuracy_score(y_test, nb_pred)
knn_acc = accuracy_score(y_test, knn_pred)
xg_acc = accuracy_score(y_test, xg_pred)

log_f1 = f1_score(y_test, log_pred, pos_label=1)
rf_f1 = f1_score(y_test, rf_pred, pos_label=1)
dt_f1 = f1_score(y_test, dt_pred, pos_label=1)
nb_f1 = f1_score(y_test, nb_pred, pos_label=1)
knn_f1 = f1_score(y_test, knn_pred, pos_label=1)
xg_f1 = f1_score(y_test, xg_pred, pos_label=1)

model_names = [
    "Logistic", "Random Forest", "Decision Tree", "Naive Bayes", "KNN", "XGBoost"
]
model_accuracy = [log_acc, rf_acc, dt_acc, nb_acc, knn_acc, xg_acc]
model_f1 = [log_f1, rf_f1, dt_f1, nb_f1, knn_f1, xg_f1]

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots()
    ax1.bar(model_names, model_accuracy)
    ax1.set_ylim(0, 1)
    ax1.set_title("Accuracy Comparison")
    ax1.set_ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots()
    ax2.bar(model_names, model_f1)
    ax2.set_ylim(0, 1)
    ax2.set_title("F1 Score (Poisonous Class)")
    ax2.set_ylabel("Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig2)


# ==========================================================
# 5️⃣ ROC Curve
# ==========================================================
st.header("📊 ROC Curves - Model Comparison")

models = {
    "Logistic": log_prob,
    "Random Forest": rf_prob,
    "Decision Tree": dt_prob,
    "Naive Bayes": nb_prob,
    "KNN": knn_prob,
    "XGBoost": xg_prob,
}

model_names = list(models.keys())

# Create 2 columns layout
for i in range(0, len(model_names), 2):
    col1, col2 = st.columns(2)

    for col, model_name in zip([col1, col2], model_names[i:i+2]):
        with col:
            prob = models[model_name]
            fpr, tpr, _ = roc_curve(y_test, prob)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax.plot([0, 1], [0, 1], linestyle="--")
            ax.set_title(model_name)
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend(loc="lower right")
            ax.tick_params(labelsize=8)

            st.pyplot(fig)


# ==========================================================
# 6️⃣ Confusion Matrix
# ==========================================================
st.header("🔎 Confusion Matrices - Model Comparison")

predictions = {
    "Logistic": log_pred,
    "Random Forest": rf_pred,
    "Decision Tree": dt_pred,
    "Naive Bayes": nb_pred,
    "KNN": knn_pred,
    "XGBoost": xg_pred,
}

model_names = list(predictions.keys())

for i in range(0, len(model_names), 2):
    col1, col2 = st.columns(2)

    for col, model_name in zip([col1, col2], model_names[i:i+2]):
        with col:
            cm = confusion_matrix(y_test, predictions[model_name])

            fig, ax = plt.subplots(figsize=(4, 4))
            im = ax.imshow(cm)

            ax.set_title(model_name)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["Edible", "Poisonous"])
            ax.set_yticklabels(["Edible", "Poisonous"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")

            # Annotate values
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, cm[i, j],
                            ha="center",
                            va="center",
                            color="white",
                            fontsize=10)

            fig.colorbar(im, ax=ax)
            st.pyplot(fig)


# ==========================================================
# 7️⃣ Feature Importance
# ==========================================================
st.header("⭐ Top 10 Feature Importance - Random Forest")

importances = rf_model.feature_importances_
feature_names = X_test.columns

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).head(10)

fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.invert_yaxis()
ax.set_xlabel("Importance Score")

st.pyplot(fig)


# ==========================================================
# 8️⃣ Interactive Prediction with Model Selection
# ==========================================================
st.header("🧪 Try Your Own Prediction")

# ---------------------------
# Model Selection Dropdown
# ---------------------------
model_options = {
    "Logistic Regression": logistic_model,
    "Random Forest": rf_model,
    "Decision Tree": decision_tree_model,
    "Naive Bayes": naive_bayes_model,
    "KNN": knn_model,
    "XGBoost": xgboost_model
}

selected_model_name = st.selectbox(
    "Select Model for Prediction",
    list(model_options.keys())
)

with st.expander("🧪 Manual Prediction - Click to Open", expanded=False):

    selected_model = model_options[selected_model_name]

    st.write(f"🔎 Using Model: **{selected_model_name}**")

    # ---------------------------
    # Dynamic Feature Inputs
    # ---------------------------
    user_inputs = {}

    cols = st.columns(3)

    for i, feature in enumerate(feature_columns):
        unique_values = sorted(df_full[feature].unique())

        with cols[i % 3]:
            user_inputs[feature] = st.selectbox(
                label=feature.replace("-", " ").title(),
                options=unique_values,
                key=f"manual_{feature}"
            )

    # ---------------------------
    # Prediction Button
    # ---------------------------
    if st.button("Predict Mushroom Type", key="manual_predict"):

        input_df = pd.DataFrame([user_inputs])

        input_encoded = pd.get_dummies(input_df)

        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        prediction = selected_model.predict(input_encoded)

        probability = None
        if hasattr(selected_model, "predict_proba"):
            probability = selected_model.predict_proba(input_encoded)[0][1]

        if prediction[0] == 1:
            if probability is not None:
                st.error(f"⚠️ Poisonous Mushroom (Confidence: {probability:.3f})")
            else:
                st.error("⚠️ Poisonous Mushroom")
        else:
            if probability is not None:
                st.success(f"✅ Edible Mushroom (Confidence: {1 - probability:.3f})")
            else:
                st.success("✅ Edible Mushroom")


# ==========================================================
# 9️⃣ Upload Test File & Evaluate Model
# ==========================================================
st.header("📂 Batch Prediction & Evaluation")

uploaded_file = st.file_uploader(
    "Upload a Test CSV File (Ensure 'class' column is present)",
    type=["csv"]
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)
    test_df.columns = test_df.columns.str.strip()

    st.write("Preview of Uploaded Data:")
    st.dataframe(test_df.head())

    # Ensure class column exists for scoring
    if "class" not in test_df.columns:
        st.error("Uploaded file must contain a 'class' column for evaluation.")
    else:

        # Separate features & target
        y_true = test_df["class"]

        # Drop same columns removed during training
        X_test_upload = test_df.drop(
            columns=["class", "odor", "gill-color", "veil-type"],
            errors="ignore"
        )

        # Encode target like training
        y_true = y_true.map({"e": 0, "p": 1})

        # One-hot encode (NO drop_first here)
        X_encoded = pd.get_dummies(X_test_upload)

        # Align with training columns
        X_encoded = X_encoded.reindex(
            columns=model_columns,
            fill_value=0
        )

        # Run prediction
        y_pred = selected_model.predict(X_encoded)

        # Try probability
        y_prob = None
        if hasattr(selected_model, "predict_proba"):
            y_prob = selected_model.predict_proba(X_encoded)[:, 1]

        # Convert predictions to labels
        y_pred_labels = ["Poisonous" if p == 1 else "Edible" for p in y_pred]

        # Show predictions
        result_df = test_df.copy()
        result_df["Predicted"] = y_pred_labels

        st.subheader("📊 Prediction Results")
        st.dataframe(result_df.head())

        # ======================================================
        # Compute Evaluation Metrics
        # ======================================================
        from sklearn.metrics import (
            accuracy_score,
            roc_auc_score,
            precision_score,
            recall_score,
            f1_score,
            matthews_corrcoef,
        )

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")
        f1 = f1_score(y_true, y_pred, average="weighted")
        mcc = matthews_corrcoef(y_true, y_pred)

        auc_score = None
        if y_prob is not None:
            auc_score = roc_auc_score(y_true, y_prob)

        st.subheader("📈 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col2.metric("Precision", f"{precision:.4f}")
        col3.metric("Recall", f"{recall:.4f}")
        col4.metric("F1 Score", f"{f1:.4f}")
        col5.metric("MCC", f"{mcc:.4f}")

        if auc_score is not None:
            col6.metric("AUC", f"{auc_score:.4f}")

        st.download_button(
            "Download Predictions",
            result_df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )

        # ======================================================
        # Confusion Matrix & ROC Curve
        # ======================================================
        st.subheader("📊 Visual Evaluation")

        col1, col2 = st.columns(2)

        # -------------------------------
        # Confusion Matrix
        # -------------------------------
        with col1:
            cm = confusion_matrix(y_true, y_pred)

            fig_cm, ax_cm = plt.subplots(figsize=(4, 4))
            im = ax_cm.imshow(cm)

            ax_cm.set_title(f"Confusion Matrix - {selected_model_name}")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            ax_cm.set_xticks([0, 1])
            ax_cm.set_yticks([0, 1])
            ax_cm.set_xticklabels(["Edible", "Poisonous"])
            ax_cm.set_yticklabels(["Edible", "Poisonous"])

            for i in range(2):
                for j in range(2):
                    ax_cm.text(j, i, cm[i, j],
                               ha="center",
                               va="center",
                               color="white",
                               fontsize=12)

            fig_cm.colorbar(im)
            st.pyplot(fig_cm)

        # -------------------------------
        # ROC Curve
        # -------------------------------
        if y_prob is not None:
            with col2:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                roc_auc = auc(fpr, tpr)

                fig_roc, ax_roc = plt.subplots(figsize=(4, 4))
                ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
                ax_roc.plot([0, 1], [0, 1], linestyle="--")

                ax_roc.set_xlabel("False Positive Rate")
                ax_roc.set_ylabel("True Positive Rate")
                ax_roc.set_title(f"ROC Curve - {selected_model_name}")
                ax_roc.legend(loc="lower right")

                st.pyplot(fig_roc)
            

