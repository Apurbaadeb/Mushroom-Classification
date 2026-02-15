# 🍄 Mushroom Classification Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A complete end-to-end **Machine Learning Classification Project** that
predicts whether a mushroom is **edible or poisonous** based on its
physical characteristics.

------------------------------------------------------------------------

## 📌 Problem Statement

Mushroom classification is a **binary classification problem** where the
objective is to determine whether a mushroom is:

-   🍄 **Edible (e)**
-   ☠️ **Poisonous (p)**

Accurate classification is critical since some poisonous mushrooms
resemble edible ones.

------------------------------------------------------------------------

## 🧠 Models Implemented

-   Logistic Regression\
-   Decision Tree\
-   K-Nearest Neighbors (KNN)\
-   Naive Bayes\
-   Random Forest\
-   XGBoost

### Evaluation Metrics

-   Accuracy\
-   Precision\
-   Recall\
-   F1 Score\
-   AUC\
-   Matthews Correlation Coefficient (MCC)\
-   ROC Curves\
-   Confusion Matrices

------------------------------------------------------------------------

## 🗂️ Project Structure

    Mushroom-Classification/
    │
    ├── mushrooms.csv
    ├── requirements.txt
    ├── Assignment 2 ML.py      # Model training script
    ├── app.py                  # Streamlit dashboard
    ├── models/                 # Saved trained models
    ├── model_columns.pkl       # Saved encoded feature structure
    ├── X_test.pkl
    ├── y_test.pkl
    └── README.md

------------------------------------------------------------------------

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository

    git clone <your-repo-url>
    cd Mushroom-Classification

### 2️⃣ Install Dependencies

    pip install -r requirements.txt

### 3️⃣ Train the Models

    python "Assignment 2 ML.py"

### 4️⃣ Run the Streamlit Dashboard

    streamlit run app.py

------------------------------------------------------------------------

## 📊 Dashboard Features

-   Model comparison (Accuracy, F1 Score)
-   ROC Curves (Grid Layout)
-   Confusion Matrices (Grid Layout)
-   Feature Importance
-   Manual Prediction (Accordion UI)
-   Batch Prediction with evaluation
-   Downloadable predictions

------------------------------------------------------------------------

## ⚙️ Machine Learning Pipeline

1.  Data Cleaning\
2.  One-Hot Encoding\
3.  Correlation-Based Feature Selection\
4.  Stratified Train-Test Split\
5.  Model Training\
6.  Evaluation\
7.  Model Serialization (Pickle)\
8.  Streamlit Deployment

------------------------------------------------------------------------

## 📌 Notes

-   Always run `Assignment 2 ML.py` before `app.py`
-   Uploaded test CSV must contain a `class` column for evaluation
-   The app aligns uploaded data with the training feature structure
    automatically

------------------------------------------------------------------------

