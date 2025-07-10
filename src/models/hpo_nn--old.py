from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from src.models.neural_net import build_nn, train_final_model
import numpy as np

def run_nn_kfold_with_hpo(X_train, y_train, X_val, X_test, y_val, y_test, k=5):
    """
    1. Hyperparameter tuning with K-Fold CV on X_train, y_train
       (try different configs, pick best based on val MAE)
    2. Train final model on full X_train with best config
    3. Evaluate on X_val and X_test
    """
    param_grid = [
        {'n_units_1': 64, 'n_units_2': 32, 'batch_size': 64, 'epochs': 20},
        {'n_units_1': 128, 'n_units_2': 64, 'batch_size': 32, 'epochs': 30},
        {'n_units_1': 64, 'n_units_2': 64, 'batch_size': 64, 'epochs': 25},
    ]

    best_mae = float('inf')
    best_params = None

    print("\nStarting hyperparameter tuning with K-Fold CV...")
    for params in param_grid:
        mae_scores = []
        print(f"Testing config: {params}")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]

            model = build_nn(X_train.shape[1], params['n_units_1'], params['n_units_2'])
            es = EarlyStopping(monitor='val_mae', patience=3, restore_best_weights=True)
            model.fit(
                X_t, y_t,
                epochs=params['epochs'],
                batch_size=params['batch_size'],
                validation_data=(X_v, y_v),
                verbose=0,
                callbacks=[es]
            )
            loss, mae = model.evaluate(X_v, y_v, verbose=0)
            mae_scores.append(mae)

        avg_mae = np.mean(mae_scores)
        print(f"Avg MAE for config: {avg_mae:.4f}")

        # External validation MAE
        model = build_nn(X_train.shape[1], params['n_units_1'], params['n_units_2'])
        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            verbose=0,
            callbacks=[EarlyStopping(monitor='val_mae', patience=3, restore_best_weights=True)]
        )
        _, val_mae = model.evaluate(X_val, y_val, verbose=0)
        print(f"Validation set MAE for config: {val_mae:.4f}")

        if val_mae < best_mae:
            best_mae = val_mae
            best_params = params

    print(f"\nBest hyperparameters found: {best_params} with Validation MAE: {best_mae:.4f}")

    # Train final model and predict
    X_val = np.array(X_val).astype(np.float32)
    X_test = np.array(X_test).astype(np.float32)

    final_model = train_final_model(X_train, y_train, X_train.shape[1], best_params)
    y_pred_val = final_model.predict(X_val).flatten()
    y_pred_test = final_model.predict(X_test).flatten()

    return y_pred_val, y_pred_test, best_params, best_mae