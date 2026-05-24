import joblib
import pandas as pd
import os

# Get root project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to saved pipeline model
MODEL_PATH = os.path.join(BASE_DIR, "models", "invoice_pipeline.pkl")


def load_pipeline():
    # Load trained pipeline (model + scaler)
    return joblib.load(MODEL_PATH)


def predict(input_data: dict):
    # Convert input dictionary to DataFrame
    df = pd.DataFrame(input_data)

    # Create feature used during training
    df["dollar_diff_ratio"] = abs(df["invoice_dollars"] - df["total_item_dollars"]) / (
        df["total_item_dollars"] + 1
    )

    # Load pipeline
    pipeline = load_pipeline()

    # Predict class (0 = normal, 1 = flagged)
    preds = pipeline.predict(df)

    # Predict probability of being flagged
    probs = pipeline.predict_proba(df)[:, 1]

    # Add results to dataframe
    df["prediction"] = preds
    df["risk_score"] = probs

    return df


if __name__ == "__main__":
    # Input must match training features (NO extra columns)
    sample_data = {
        "invoice_quantity": [100],
        "invoice_dollars": [5000],
        "Freight": [200],
        "total_item_quantity": [95],
        "total_item_dollars": [4800],
        "avg_receiving_delay": [8]
    }

    result = predict(sample_data)
    print(result)