
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

app = Flask(__name__)

# --------------------------------------------------
# File paths
# --------------------------------------------------
BASE_DIR = Path(__file__).parent

MODEL_PATH = BASE_DIR / "AdaBoost_final_defect_prediction_model.pkl"
CSV_PATH = BASE_DIR / "DillibabuSarva_DefectDataset.csv"

# --------------------------------------------------
# Load trained model and deployment metadata
# --------------------------------------------------
with open(MODEL_PATH, "rb") as file:
    deployment_data = pickle.load(file)

if isinstance(deployment_data, dict):
    model = deployment_data["model"]
    features = deployment_data.get(
        "features",
        [
            "nosi", "dit", "cbo", "rfc", "maxNestedBlocks",
            "uniqueWordsQty", "assignmentsQty", "numbersQty",
            "tryCatchQty", "parenthesizedExpsQty"
        ]
    )
    medians = deployment_data.get("medians", {})
else:
    model = deployment_data
    features = [
        "nosi", "dit", "cbo", "rfc", "maxNestedBlocks",
        "uniqueWordsQty", "assignmentsQty", "numbersQty",
        "tryCatchQty", "parenthesizedExpsQty"
    ]
    medians = {}

# --------------------------------------------------
# Load dataset and calculate actual minimum/maximum
# --------------------------------------------------
df = pd.read_csv(CSV_PATH)

# Verify that all required features exist
missing_features = [col for col in features if col not in df.columns]

if missing_features:
    raise ValueError(
        f"Missing features in CSV dataset: {missing_features}"
    )

# Actual dataset ranges
lower = {col: float(df[col].min()) for col in features}
upper = {col: float(df[col].max()) for col in features}

# Use dataset medians as default values
for col in features:
    if col not in medians or not np.isfinite(float(medians[col])):
        medians[col] = float(df[col].median())

# Human-readable feature names
expanded_names = {
    "nosi": "NOSI (Number of Static Invocations)",
    "dit": "DIT (Depth of Inheritance Tree)",
    "cbo": "CBO (Coupling Between Objects)",
    "rfc": "RFC (Response For a Class)",
    "maxNestedBlocks": "Maximum Nested Blocks",
    "uniqueWordsQty": "Unique Words Quantity",
    "assignmentsQty": "Assignments Quantity",
    "numbersQty": "Numbers Quantity",
    "tryCatchQty": "Try/Catch Quantity",
    "parenthesizedExpsQty": "Parenthesized Expressions"
}


# --------------------------------------------------
# Home page
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------
# Prediction form
# --------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template(
            "predict.html",
            features=features,
            lower=lower,
            upper=upper,
            medians=medians,
            expanded_names=expanded_names
        )

    try:
        values = {}

        for feature in features:
            raw_value = request.form.get(feature, "").strip()

            if not raw_value:
                raise ValueError(
                    f"Please enter a value for {expanded_names.get(feature, feature)}."
                )

            value = float(raw_value)

            if not np.isfinite(value):
                raise ValueError(
                    f"{feature} must be a finite number."
                )

            # Check actual dataset limits
            if value < lower[feature] or value > upper[feature]:
                raise ValueError(
                    f"{expanded_names.get(feature, feature)} must be "
                    f"between {lower[feature]:g} and {upper[feature]:g}."
                )

            values[feature] = value

        # Keep feature order identical to model training
        input_data = pd.DataFrame(
            [[values[col] for col in features]],
            columns=features
        )

        prediction = model.predict(input_data)[0]

        return render_template(
            "result.html",
            prediction=str(prediction),
            input_values=values
        )

    except (ValueError, TypeError) as error:
        return render_template(
            "predict.html",
            features=features,
            lower=lower,
            upper=upper,
            medians=medians,
            expanded_names=expanded_names,
            error=str(error)
        ), 400


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
