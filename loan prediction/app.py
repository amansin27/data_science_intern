# app.py
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load saved model and encoders
model = joblib.load("loan_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")
num_cols, cat_cols = joblib.load("feature_columns.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input
        data = {key: request.form[key] for key in request.form}

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Convert numerical fields properly
        for col in num_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Predict
        pred = model.predict(df)[0]
        result = label_encoder.inverse_transform([pred])[0]

        if result == 'Y':
            msg = "✅ Loan Approved!"
        else:
            msg = "❌ Loan Not Approved!"

        return render_template("index.html", prediction_text=msg)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
