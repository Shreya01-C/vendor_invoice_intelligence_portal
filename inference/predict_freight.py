import joblib
import pandas as pd
import os


# project root path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "invoice_pipeline.pkl")


def load_model():
    """load trained pipeline"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars",
    "avg_receiving_delay",
    "dollar_diff_ratio"
]


def prepare_input(input_data):
    """convert input + create features"""
    df = pd.DataFrame(input_data)

    # same feature logic as training
    df["dollar_diff_ratio"] = abs(
        df["invoice_dollars"] - df["total_item_dollars"]
    ) / (df["total_item_dollars"] + 1)

    return df[FEATURES]


def predict(input_data):
    """run prediction"""
    model = load_model()
    df = prepare_input(input_data)

    preds = model.predict(df)
    probs = model.predict_proba(df)[:, 1]

    df["prediction"] = preds
    df["risk_score"] = probs

    return df


if __name__ == "__main__":
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