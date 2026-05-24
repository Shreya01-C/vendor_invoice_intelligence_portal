import joblib
from pathlib import Path

# Import custom functions
from data_preprocessing import load_vendor_invoice_data, prepare_features, split_data
from model_evaluation import (
    train_linear_regression,
    train_decision_tree,
    train_random_forest,
    evaluate_model
)

def main():
    # -------------------- DATABASE PATH --------------------
    # ⚠️ Use correct Windows path (your current one is Mac-style)
    db_path = r"C:\Users\navya\OneDrive\Desktop\ml_project\data\inventory.db"

    # -------------------- MODEL DIRECTORY --------------------
    # Create folder to store model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # -------------------- LOAD DATA --------------------
    # Load dataset from database
    df = load_vendor_invoice_data(db_path)

    # -------------------- PREPARE DATA --------------------
    # Split into features (X) and target (y)
    X, y = prepare_features(df)

    # Train-test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # -------------------- TRAIN MODELS --------------------
    # Train all models
    lr_model = train_linear_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)

    # -------------------- EVALUATE MODELS --------------------
    # Store evaluation results
    results = []

    results.append(evaluate_model(lr_model, X_test, y_test, "Linear Regression"))
    results.append(evaluate_model(dt_model, X_test, y_test, "Decision Tree Regression"))
    results.append(evaluate_model(rf_model, X_test, y_test, "Random Forest Regression"))

    # -------------------- SELECT BEST MODEL --------------------
    # Select model with lowest MAE
    best_model_info = min(results, key=lambda x: x["MAE"])

    # ⚠️ FIX: key is "model", NOT "model_name"
    best_model_name = best_model_info["model"]

    # -------------------- MODEL MAPPING --------------------
    # ⚠️ FIX: names must EXACTLY match evaluation names
    models = {
        "Linear Regression": lr_model,
        "Decision Tree Regression": dt_model,
        "Random Forest Regression": rf_model
    }

    # Get best model
    best_model = models[best_model_name]

    # -------------------- SAVE MODEL --------------------
    model_path = model_dir / "predict_freight_model.pkl"
    joblib.dump(best_model, model_path)

    # -------------------- OUTPUT --------------------
    print("\nModel Results:")
    for res in results:
        print(res)

    print(f"\nBest model saved: {best_model_name}")
    print(f"Model path: {model_path}")


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    main()