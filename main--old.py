import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.preprocessing import load_and_clean_data, get_feature_target_split, get_preprocessor
from src.models.baseline import compare_all_baselines
from src.models.neural_net import build_nn
from src.evaluation import evaluate_model
from src.models.hpo_nn import run_nn_kfold_with_hpo

def main():
    print("\nStarting Flight Delay Prediction Pipeline...\n")

    # 1. Load and preprocess data
    df = load_and_clean_data('data/flights_sample_100k.csv')
    X, y = get_feature_target_split(df)

    # Split 60% train, 20% validation, 20% test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2

    print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# 3. Preprocess numeric columns
    numeric_features = [
        'CRS_DEP_HOUR', 'CRS_ARR_HOUR', 'TAXI_OUT', 'TAXI_IN',
        'AIR_TIME', 'DISTANCE', 'DAY_OF_WEEK',
        'DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
        'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT'
    ]
    preprocessor = get_preprocessor(numeric_features)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    X_train_processed = np.array(X_train_processed).astype(np.float32)
    X_val_processed = np.array(X_val_processed).astype(np.float32)
    X_test_processed = np.array(X_test_processed).astype(np.float32)

# 4. Train NN with HPO using validation set
    y_pred_val, y_pred_test, best_params, best_val_mae = run_nn_kfold_with_hpo(
        X_train_processed, y_train.values,
        X_val_processed, y_val.values,
        X_test_processed, y_test.values
    )

    print(f"\nBest Hyperparameters: {best_params}")
    print(f"Validation MAE with best params: {best_val_mae:.4f}")

# 5. Evaluate on validation and test
    evaluate_model(y_val, y_pred_val, label="Validation Set")
    evaluate_model(y_test, y_pred_test, label="Test Set")

# 6. Baseline comparison (only on test set)
    compare_all_baselines(X_train, y_train, X_test, y_test, y_pred_test)

    print("\n Done! All models trained, validated, and compared.\n")


if __name__ == "__main__":
    main()
