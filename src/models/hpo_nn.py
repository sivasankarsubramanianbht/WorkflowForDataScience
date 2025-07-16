# --- hpo_nn.py ---
from sklearn.model_selection import KFold
import numpy as np
from .neural_net import build_nn
from tensorflow.keras.callbacks import EarlyStopping
from .neural_net import train_final_model

def run_nn_kfold_with_hpo(X_train, y_train, X_val, X_test, y_val, y_test, k=5):
    param_grid = [
        {'n_units_1': 64, 'n_units_2': 32, 'batch_size': 64, 'epochs': 20},
        {'n_units_1': 128, 'n_units_2': 64, 'batch_size': 32, 'epochs': 30},
        {'n_units_1': 64, 'n_units_2': 64, 'batch_size': 64, 'epochs': 25},
    ]

    best_mae = float('inf')
    best_params = None

    for params in param_grid:
        mae_scores = []
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(X_train):
            X_t, X_v = X_train[train_idx], X_train[val_idx]
            y_t, y_v = y_train[train_idx], y_train[val_idx]

            model = build_nn(X_train.shape[1], params['n_units_1'], params['n_units_2'])
            model.fit(X_t, y_t, epochs=params['epochs'], batch_size=params['batch_size'],
                      validation_data=(X_v, y_v), verbose=0,
                      callbacks=[EarlyStopping(monitor='val_mae', patience=3, restore_best_weights=True)])
            _, mae = model.evaluate(X_v, y_v, verbose=0)
            mae_scores.append(mae)

        avg_mae = np.mean(mae_scores)

        model = build_nn(X_train.shape[1], params['n_units_1'], params['n_units_2'])
        model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'],
                  validation_data=(X_val, y_val), verbose=0,
                  callbacks=[EarlyStopping(monitor='val_mae', patience=3, restore_best_weights=True)])
        _, val_mae = model.evaluate(X_val, y_val, verbose=0)

        if val_mae < best_mae:
            best_mae = val_mae
            best_params = params

    #final_model = build_nn(X_train.shape[1], best_params['n_units_1'], best_params['n_units_2'])
    #final_model.fit(X_train, y_train, epochs=best_params['epochs'], batch_size=best_params['batch_size'],
                    #verbose=0, callbacks=[EarlyStopping(monitor='val_mae', patience=3, restore_best_weights=True)])
    final_model,history = train_final_model(X_train, y_train, X_val, y_val, X_train.shape[1], best_params)

    y_pred_val = final_model.predict(X_val).flatten()
    y_pred_test = final_model.predict(X_test).flatten()

    return y_pred_val, y_pred_test, best_params, best_mae,history
