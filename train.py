import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\user\Desktop\mushrooms.csv")

# Separate target
y = df["class"]
X = df.drop(columns=["class"])

# Encode categorical features
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Train models
logistic_model = LogisticRegression(max_iter=1000)
rf_model = RandomForestClassifier()

logistic_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Save everything
pickle.dump(logistic_model, open("logistic_model.pkl", "wb"))
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))
pickle.dump(X_test, open("X_test.pkl", "wb"))
pickle.dump(y_test, open("y_test.pkl", "wb"))

print("✅ Models saved successfully!")
