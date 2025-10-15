# model_build.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("loan predict.csv")

# Target column
target_col = "Loan_Status"

# Split X and y
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target
le = LabelEncoder()
y = le.fit_transform(y)

# Drop ID if present
if "Loan_ID" in X.columns:
    X = X.drop(columns=["Loan_ID"])

# Identify numerical and categorical features
num_cols = X.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

# Preprocessing
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
model.fit(X_train, y_train)
acc = model.score(X_test, y_test)
print(f"Model trained successfully with accuracy: {acc:.3f}")

# Save model & label encoder
joblib.dump(model, "loan_model.pkl")
joblib.dump(le, "label_encoder.pkl")
joblib.dump((num_cols, cat_cols), "feature_columns.pkl")

print("Model and encoders saved successfully!")
