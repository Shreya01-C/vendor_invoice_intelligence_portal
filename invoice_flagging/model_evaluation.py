# -------------------- IMPORT LIBRARIES --------------------

# Machine Learning models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Numpy for RMSE calculation (ensures compatibility with older sklearn versions)
import numpy as np


# -------------------- TRAINING FUNCTIONS --------------------

def train_linear_regression(X_train, y_train):
    """
    Train and return a Linear Regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train):
    """
    Train and return a Decision Tree Regressor
    """
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    """
    Train and return a Random Forest Regressor
    """
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


# -------------------- EVALUATION FUNCTION --------------------

def evaluate_model(model, X, y, model_name):
    """
    Evaluate a regression model using standard metrics

    Parameters:
    ----------
    model : trained model
    X     : input features
    y     : actual target values
    model_name : str

    Returns:
    -------
    dict : containing model name and evaluation metrics
    """

    # Generate predictions
    preds = model.predict(X)

    # Calculate evaluation metrics
    mae = mean_absolute_error(y, preds)           # Average absolute error
    rmse = np.sqrt(mean_squared_error(y, preds))  # Penalizes large errors more
    r2 = r2_score(y, preds)                       # Model fit quality (0–1 typically)

    # Return structured results
    return {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }