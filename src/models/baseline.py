import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.evaluation import evaluate_model as evaluate

def baseline_mean(X_train, y_train, X_test, y_test):
    """Naive baseline: predict global mean."""
    y_pred = np.full_like(y_test, fill_value=np.mean(y_train), dtype=float)
    mae, rmse, r2 = evaluate(y_test, y_pred, label="Naive Mean")
    return mae, rmse, r2


#def baseline_median(X_train, y_train, X_test, y_test):
    """Naive baseline: predict global median."""
    #dummy = DummyRegressor(strategy='median')
    #dummy.fit(X_train, y_train)
    #y_pred = dummy.predict(X_test)
    #mae, rmse, r2 = evaluate(y_test, y_pred, label="Naive Median")
    #return mae, rmse, r2


#def baseline_zero(X_test, y_test):
    """Naive baseline: predict zero delay (assume on time)."""
    #y_pred = np.zeros(len(y_test))
    #mae, rmse, r2 = evaluate(y_test, y_pred, label="Naive Zero")
    #return mae, rmse, r2


def compare_all_baselines(X_train, y_train, X_test, y_test, y_pred_nn=None):
    """Compare all baselines + neural net (optional). Print results."""
    results = []

    mae_mean, rmse_mean, r2_mean = baseline_mean(X_train, y_train, X_test, y_test)
    #mae_median, rmse_median, r2_median = baseline_median(X_train, y_train, X_test, y_test)
    #mae_zero, rmse_zero, r2_zero = baseline_zero(X_test, y_test)
    mae_nn, rmse_nn, r2_nn = evaluate(y_test, y_pred_nn, label="Neural Network")

    df = pd.DataFrame({
        'Model': ['Naive Mean', 'Neural Network'],
        'MAE': [mae_mean, mae_nn],
        'RMSE': [rmse_mean, rmse_nn],
        'R²': [r2_mean, r2_nn]
    })
    print("\n Model Comparison:")
    print(df.round(3))
    return df
